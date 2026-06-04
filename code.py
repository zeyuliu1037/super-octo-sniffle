"""Transformer model for Deep Cross Attention."""

# pylint: disable=invalid-name

import dataclasses
import math
from typing import Optional, Tuple
from flax import linen as nn
import jax
import jax.numpy as jnp
from nanodo import fsdp
from nanodo import model as nanodo_model

@dataclasses.dataclass
class DoConfig(nanodo_model.DoConfig):
  """Extended Hyper-parameters for DCA."""

  experimental_model: str = ''
  grn_k: int = -1
  logits_via_embedding: bool = False
  qkv_dim: Optional[int] = 256
  mlp_dim: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
  bias_init: nn.initializers.Initializer = nn.initializers.normal(stddev=1e-6)
  remat: bool = False
  stop_gradient_compression: bool = True
  per_dimension_compression: bool = True
  # Post-projection per-head DCA: independent cross-layer mixing per axis.
  dca_mix_q: bool = False
  dca_mix_k: bool = False
  dca_mix_v: bool = False
  # MLP read mode for the post-proj path. One of:
  #   none | grn_pool | grn_pool_res_in | grn_pool_res_out
  dca_mlp_read: str = "none"
  # Number of (first) heads that mix across depth; shared across Q/K/V.
  #   -1 (default) => all H heads; 0 => no heads mix (axis passes through).
  dca_mix_heads_k: int = -1
  # QKV depth-gate granularity: True => per-head gate; False => one shared
  # per-slot scale across heads (the original D-space InputSelector gate).
  dca_grn_per_head: bool = True
  # QKV depth-gate: True => relu(logit + learnable bias) (sum-init);
  # False => softmax over depth slots, no bias (mean-init).
  dca_grn_bias: bool = True
  # DCA GRN hard source selection. 0 keeps current behavior; integer k>=1
  # keeps exactly k source slots per token/head; 0<x<1 keeps ceil(K*x) slots.
  # Bias-gated selectors rank layers after forming relu(logit + bias), then
  # apply the same layer mask to the full coefficient.
  dca_grn_topk: float = 0.0


class MlpBlock(nn.Module):
  """Transformer MLP/FFN block with dropout."""

  config: DoConfig

  @nn.compact
  def __call__(self, inputs, deterministic: Optional[bool] = None):
    config = self.config
    actual_emb_dim = config.D
    actual_mlp_dim = config.mlp_dim

    if deterministic is None:
      deterministic = config.deterministic or not self.has_rng('dropout')

    x = nn.Dense(
        actual_mlp_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
    )(inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
    x = nn.Dense(
        actual_emb_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
    )(x)
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
    return x


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer."""

  config: DoConfig

  @nn.compact
  def __call__(
      self,
      inputs_q,
      inputs_k=None,
      inputs_v=None,
      decoder_mask=None,
      encoder_decoder_mask=None,
      deterministic: Optional[bool] = None,
  ):
    config = self.config

    if deterministic is None:
      deterministic = config.deterministic or not self.has_rng('dropout')

    assert inputs_q.ndim == 3
    x_q = nn.LayerNorm(dtype=config.dtype)(inputs_q)
    if inputs_k is not None:
      assert inputs_v is not None
      x_k = nn.LayerNorm(dtype=config.dtype)(inputs_k)
      x_v = nn.LayerNorm(dtype=config.dtype)(inputs_v)
    else:
      x_k = x_v = x_q

    x = nn.MultiHeadAttention(
        num_heads=config.H,
        dtype=config.dtype,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=deterministic,
        decode=config.decode,
    )(x_q, x_k, x_v, mask=decoder_mask)
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)

    # MLP block.
    z = nn.LayerNorm(dtype=config.dtype)(inputs_q + x)
    z = MlpBlock(config=config)(z, deterministic=deterministic)

    return x + z


def _resolve_grn_keep_k(cfg, num_inputs: int) -> int:
  """Number of depth slots to keep for dynamic GRN hard selection."""
  topk = float(cfg.dca_grn_topk)
  if topk == 0.0:
    return num_inputs
  assert topk > 0.0, "dca_grn_topk must be non-negative"
  if topk < 1.0:
    return max(1, min(math.ceil(num_inputs * topk), num_inputs))
  assert topk.is_integer(), (
      "dca_grn_topk >= 1 must be an integer; use 0<x<1 for percentage"
  )
  return min(int(topk), num_inputs)


def _exact_topk_mask(scores: jax.Array, k: int, axis: int) -> jax.Array:
  """Boolean mask with exactly k selected entries along `axis`.

  Threshold-based masks can over-select on ties, which are common when gates are
  ReLU-initialized to equal values. `top_k` indices plus one-hot gives exact-k
  semantics while staying dense/XLA-friendly.
  """
  K = scores.shape[axis]
  if k >= K:
    return jnp.ones(scores.shape, dtype=jnp.bool_)
  moved = jnp.moveaxis(scores, axis, -1)
  _, indices = jax.lax.top_k(moved, int(k))
  mask = jax.nn.one_hot(indices, K, dtype=jnp.int32).sum(axis=-2) > 0
  return jnp.moveaxis(mask, -1, axis)


def _grn_keep_mask(scores_BxLxKxAny: jax.Array, cfg) -> Optional[jax.Array]:
  """Optional exact top-k mask over the K/source-layer axis."""
  k = _resolve_grn_keep_k(cfg, scores_BxLxKxAny.shape[2])
  if k >= scores_BxLxKxAny.shape[2]:
    return None
  return _exact_topk_mask(scores_BxLxKxAny, k, axis=2)


class InputSelector(nn.Module):
  """Input selector for TransformerLM with DCA support."""

  config: DoConfig

  @nn.compact
  def __call__(self, inputs: jax.Array):
    """Input selector for TransformerLM with DCA support.

    Args:
      inputs: Input array of shape (Batch, Seq, NumInputs, Dim).

    Returns:
      Output array of shape (Batch, Seq, Dim).
    """
    config = self.config
    Dim = config.D

    if not config.experimental_model:
      return inputs[:, :, -1, :]

    num_inputs = inputs.shape[-2]


    # Create additional parameters used to weight the inputs.
    partition_fn = (
        nn.with_partitioning if config.fsdp_enabled else lambda x, y: x
    )
    query = self.param(
        'query',
        partition_fn(nn.initializers.zeros, ('data',)),
        (Dim,),
    )
    # Initialize bias to ones to ensure that the implementation is equivalent
    # to the baseline at initialization.
    bias = self.param(
        'bias',
        partition_fn(nn.initializers.ones, (None, 'data')),
        (num_inputs, Dim),
    )

    # Shared LayerNorm across inputs to match original parameter structure.
    # (Batch, Seq, NumInputs, Dim)
    keys = nn.LayerNorm(dtype=config.dtype)(inputs)
    # (Batch, Seq, NumInputs)
    logits = jnp.einsum('...li,i -> ...l', keys, query)
    # Compute weights which depend on the inputs.
    # (Batch, Seq, NumInputs, Dim)
    w = nn.relu(logits[..., None] + bias[None, None, :, :])
    keep = _grn_keep_mask(w.mean(axis=-1), config)
    if keep is not None:
      w = w * keep[..., None]
    # Compute weighted sum of inputs.
    y = jnp.einsum('...li, ...li -> ...i', inputs, w)
    return y


class HeadwiseInputSelector(nn.Module):
  """Depth selector over a stack `(B, L, K, Hp, Dh)` -> `(B, L, Hp, Dh)`.

  Aggregates the K depth slots with a gate controlled by two config flags:

    cfg.dca_grn_per_head : True  -> per-head gate (logit per (token, slot, head))
                           False -> one shared per-slot scale across heads
    cfg.dca_grn_bias     : True  -> coeff = relu(logit + learnable bias) (sum init)
                           False -> coeff = softmax over K(logit), no bias (mean init)

  Defaults (True, True) are the original per-head, bias-inside-relu gate: at init
  (query=0, bias=1) the output is the sum over K. The weighted sum always uses the
  raw inputs (not the normalized keys).
  """

  config: DoConfig

  @nn.compact
  def __call__(self, inputs: jax.Array):
    config = self.config
    K = inputs.shape[-3]
    Hp = inputs.shape[-2]
    Dh = inputs.shape[-1]

    partition_fn = (
        nn.with_partitioning if config.fsdp_enabled else lambda x, y: x
    )
    query = self.param(
        'query',
        partition_fn(nn.initializers.zeros, (None, 'data')),
        (Hp, Dh),
    )

    # LayerNorm over Dh (no bias); only feeds the gate.
    keys = nn.LayerNorm(dtype=config.dtype, use_bias=False)(inputs)
    if config.dca_grn_per_head:
      logits = jnp.einsum('blkhd,hd->blkh', keys, query)   # (B,L,K,Hp)
      logits_b = logits[..., None]                          # (B,L,K,Hp,1)
    else:
      logits = jnp.einsum('blkhd,hd->blk', keys, query)    # (B,L,K) shared
      logits_b = logits[..., None, None]                   # (B,L,K,1,1)

    if config.dca_grn_bias:
      bias = self.param(
          'bias',
          partition_fn(nn.initializers.ones, (None, None, 'data')),
          (K, Hp, Dh),
      )
      coeff = nn.relu(logits_b + bias[None, None, :, :, :])  # (B,L,K,Hp,Dh)
      keep = _grn_keep_mask(coeff.mean(axis=-1), config)
      if keep is not None:
        coeff = coeff * keep[..., None]
    else:
      keep = _grn_keep_mask(logits, config)
      if keep is not None:
        logits = jnp.where(keep, logits, jnp.finfo(jnp.float32).min)
      w = jax.nn.softmax(logits, axis=2)                    # over the K axis
      coeff = w[..., None] if config.dca_grn_per_head else w[..., None, None]

    # Multiply-then-sum over K (broadcasts coeff's size-1 axes in the softmax
    # path; equals the old einsum exactly for the full-coeff bias path).
    return jnp.sum(inputs * coeff, axis=2)                   # (B, L, Hp, Dh)


_DCA_MLP_READ_MODES = (
    'none', 'grn_pool', 'grn_pool_res_in', 'grn_pool_res_out'
)


def _resolve_km(cfg) -> int:
  """Number of (first) heads that mix across depth (shared across Q/K/V).

  `dca_mix_heads_k < 0` means all `H` heads (the default / current behavior);
  otherwise clamp to `[0, H]`.
  """
  return cfg.H if cfg.dca_mix_heads_k < 0 else min(cfg.dca_mix_heads_k, cfg.H)


def _axis_mixes(cfg, flag: bool) -> bool:
  """Whether an axis actually mixes across depth: enabled AND km > 0."""
  return bool(flag) and _resolve_km(cfg) > 0


class DcaPostProjBlock(nn.Module):
  """Post-projection per-head DCA block (standard pre-norm residual stream).

  Projects Q/K/V per head from the current hidden state, mixes each *enabled*
  axis across past layers with a `HeadwiseInputSelector`, runs causal attention,
  then a pre-FF MLP. Returns `(block_out, (q_store, k_store, v_store, mlp_slot))`.
  Each axis store is the `km`-head pre-mix slice `(B, L, km, Dh)` the decoder
  keeps for later layers, or `None` when the axis is unmixed (`dca_mix_*` off or
  `km == 0`). The stored `q` slice is already scaled by `1/sqrt(Dh)`, so the
  decoder must feed it back verbatim and every stacked slot carries uniform
  scaling.

  Precondition: for each *mixed* axis the caller must pass exactly
  `layer_index` past entries in `mem[axis]`, so `HeadwiseInputSelector`'s depth
  `K` (and thus its bias shape) is identical between `init` and every `apply`.
  """

  config: DoConfig
  layer_index: int

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array, mem: dict):
    cfg = self.config
    H = cfg.H
    assert cfg.D % cfg.H == 0, f"D {cfg.D} not divisible by H {cfg.H}"
    Dh = cfg.D // cfg.H
    assert cfg.dca_mlp_read in _DCA_MLP_READ_MODES, (
        f"unknown dca_mlp_read: {cfg.dca_mlp_read!r}"
    )

    km = _resolve_km(cfg)

    # A mixed axis reads its whole past list; require the decoder-maintained
    # invariant len(mem[axis]) == layer_index so the per-layer bias shape stays
    # consistent across init/apply (see class docstring).
    for _axis in ('q', 'k', 'v'):
      if _axis_mixes(cfg, getattr(cfg, f'dca_mix_{_axis}')):
        assert len(mem[_axis]) == self.layer_index, (
            f"{_axis}: expected {self.layer_index} past entries, "
            f"got {len(mem[_axis])}"
        )

    if cfg.dca_mlp_read != 'none':
      assert len(mem['mlp']) == self.layer_index, (
          f"mlp: expected {self.layer_index} past entries, "
          f"got {len(mem['mlp'])}"
      )

    def proj(name):
      return nn.DenseGeneral(
          features=(H, Dh),
          axis=-1,
          use_bias=False,
          kernel_init=cfg.kernel_init,
          dtype=cfg.dtype,
          name=name,
      )

    x_norm = nn.LayerNorm(
        dtype=cfg.dtype, use_bias=False, name='pre_attn_norm'
    )(x_BxLxD)
    q = proj('query')(x_norm) / (Dh ** 0.5)
    k = proj('key')(x_norm)
    v = proj('value')(x_norm)

    def apply_axis(name, projection, past_list, flag):
      # Returns (full projection for attention, slice to store or None).
      # The first `km` heads mix across depth; the rest pass through unchanged
      # into the same multi-head attention.
      if not _axis_mixes(cfg, flag):
        return projection, None
      head_slice = projection[:, :, :km, :]            # (B, L, km, Dh) mixed
      rest = projection[:, :, km:, :]                  # (B, L, H-km, Dh) passthrough
      stacked = jnp.stack([*past_list, head_slice], axis=2)  # (B, L, K, km, Dh)
      mixed = HeadwiseInputSelector(cfg, name=f'sel_{name}')(stacked)  # (B,L,km,Dh)
      full = jnp.concatenate([mixed, rest], axis=2)     # (B, L, H, Dh)
      return full, head_slice                           # store the PRE-mix slice

    q_m, q_store = apply_axis('q', q, mem['q'], cfg.dca_mix_q)
    k_m, k_store = apply_axis('k', k, mem['k'], cfg.dca_mix_k)
    v_m, v_store = apply_axis('v', v, mem['v'], cfg.dca_mix_v)

    L = x_BxLxD.shape[1]
    logits = jnp.einsum('bqhd,bkhd->bhqk', q_m, k_m).astype(jnp.float32)
    mask = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(logits, axis=-1).astype(cfg.dtype)
    attn = jnp.einsum('bhqk,bkhd->bqhd', weights, v_m)
    a = nn.DenseGeneral(
        features=cfg.D,
        axis=(-2, -1),
        use_bias=False,
        kernel_init=cfg.kernel_init,
        dtype=cfg.dtype,
        name='attn_out_proj',
    )(attn)

    x_post = x_BxLxD + a

    # MLP read mode. `pool` is the existing D-space InputSelector (1 output)
    # over [history | current slot]; `a` is the attn output. For
    # grn_pool_res_out the pre-attn input x_BxLxD is added to the pool OUTSIDE
    # the norm.
    mode = cfg.dca_mlp_read
    norm = nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name='pre_ff_norm')
    if mode == 'none':
      pre_ff = norm(x_post)
      slot = None
    else:
      cur = x_post if mode == 'grn_pool_res_in' else a
      stacked = jnp.stack([*mem['mlp'], cur], axis=2)  # (B, L, K, D)
      pooled = InputSelector(config=cfg, name='mlp_pool')(stacked)
      pre = pooled + x_BxLxD if mode == 'grn_pool_res_out' else pooled
      pre_ff = norm(pre)
      slot = cur
    ff = MlpBlock(config=cfg, name='mlp')(pre_ff)
    out = x_post + ff
    return out, (q_store, k_store, v_store, slot)


class DcaPostProjDecoder(nn.Module):
  """Decoder for post-projection per-head DCA.

  Threads per-axis memory as separate Python lists (one per enabled q/k/v axis,
  plus one for the MLP read slot): each mixed axis stores its pre-mix `km`-head
  slice `(B, L, km, Dh)` per layer (km = `_resolve_km(cfg)`); the MLP read modes
  store a D-space slot. No final pool — returns the last hidden state. Builds its
  own causal mask inside the block, so `decoder_mask` is ignored.
  """

  config: DoConfig

  @nn.compact
  def __call__(
      self,
      y: jax.Array,
      decoder_mask=None,
      encoder_decoder_mask=None,
      deterministic=None,
  ):
    del decoder_mask, encoder_decoder_mask, deterministic
    cfg = self.config
    last = y
    mem = {'q': [], 'k': [], 'v': [], 'mlp': []}
    for lyr in range(cfg.N):
      block = DcaPostProjBlock(
          config=cfg, layer_index=lyr, name=f'dca_pp_block_{lyr}'
      )
      last, (q, k, v, slot) = block(last, mem)
      mem = dict(mem)
      if _axis_mixes(cfg, cfg.dca_mix_q):
        mem['q'] = mem['q'] + [q]
      if _axis_mixes(cfg, cfg.dca_mix_k):
        mem['k'] = mem['k'] + [k]
      if _axis_mixes(cfg, cfg.dca_mix_v):
        mem['v'] = mem['v'] + [v]
      if cfg.dca_mlp_read != 'none':
        assert slot is not None  # block guarantees a slot for non-"none" reads
        mem['mlp'] = mem['mlp'] + [slot]
    return last


class DefaultDecoder(nn.Module):
  """Default Decoder for TransformerLM."""

  config: DoConfig

  @nn.compact
  def __call__(
      self,
      y: jax.Array,
      decoder_mask: Optional[jax.Array] = None,
      encoder_decoder_mask: Optional[jax.Array] = None,
      deterministic: Optional[bool] = None,
  ):
    cfg = self.config

    if deterministic is None:
      deterministic = cfg.deterministic or not self.has_rng('dropout')

    block = EncoderDecoder1DBlock
    if cfg.remat:
      block = nn.remat(block, static_argnums=6)

    for lyr in range(cfg.N):
      y_block = block(config=cfg, name=f'encoderdecoderblock_{lyr}')(
          y,
          y,
          y,
          decoder_mask,
          encoder_decoder_mask,
          deterministic,
      )
      y = y + y_block
    return y


@dataclasses.dataclass
class CompressionState:
  """State for incremental compression."""

  s: Optional[jax.Array] = None
  v: Optional[jax.Array] = None


def make_identity_preserving_init(target_pairs, large_val=10.0):
  """Factory for identity-preserving initializers.

  Args:
    target_pairs: List of (row, col) tuples where we want high probability.
    large_val: The value to set at target positions.

  Returns:
    An initializer function.
  """

  def init_fn(key, shape, dtype=jnp.float32):
    del key
    K_init = jnp.ones(shape, dtype=dtype) * -large_val
    for r, c in target_pairs:
      # Handle both 2D (k+1, k) and 3D (1, k, k+1) for attention bias
      if len(shape) == 2:
        K_init = K_init.at[r, c].set(large_val)
      elif len(shape) == 3:
        K_init = K_init.at[:, r, c].set(large_val)
    return K_init

  return init_fn


class IncrementalCompression(nn.Module):
  """Base class for incremental compression."""

  config: DoConfig
  k: int

  def __call__(
      self, layer_inputs: jax.Array, state: CompressionState
  ) -> Tuple[Optional[jax.Array], CompressionState]:
    raise NotImplementedError


class MiddleSumCompression(IncrementalCompression):
  """Middle sum compression."""

  def __call__(
      self, layer_inputs: jax.Array, state: CompressionState
  ) -> Tuple[Optional[jax.Array], CompressionState]:
    k = self.k
    num_inputs = layer_inputs.shape[-2]
    if num_inputs == k:
      return None, state
    elif num_inputs == k + 1:
      dtype = self.config.dtype
      if k == 1:
        mat = jnp.ones((1, 2, 1), dtype=dtype)
      else:
        eye = jnp.eye(k, dtype=dtype)
        rows = [eye[0:1], eye[1:2], eye[1:2]]
        if k > 2:
          rows.append(eye[2:])
        mat = jnp.concatenate(rows, axis=0)
        mat = mat[None, ...]  # Add Dim dimension: (1, k+1, k)
      return mat, state
    else:
      raise ValueError(f'Unexpected number of inputs: {num_inputs}')


class DcaIncrementalDecoder(nn.Module):
  """DCA Decoder with Incremental Compression for TransformerLM."""

  config: DoConfig

  @nn.compact
  def __call__(
      self,
      y: jax.Array,
      decoder_mask: Optional[jax.Array] = None,
      encoder_decoder_mask: Optional[jax.Array] = None,
      deterministic: Optional[bool] = None,
  ):
    cfg = self.config
    y_list = [y]
    k = cfg.grn_k

    if deterministic is None:
      deterministic = cfg.deterministic or not self.has_rng('dropout')

    block = EncoderDecoder1DBlock
    if cfg.remat:
      block = nn.remat(block, static_argnums=6)

    state = CompressionState()
    layer_inputs = None

    for lyr in range(cfg.N):

      compression = MiddleSumCompression(
          config=cfg, k=k, name=f'compression_{lyr}'
      )

      if len(y_list) <= k:
        layer_inputs = jnp.stack(y_list[:k], axis=-2)
      if len(y_list) == k:
        _, state = compression(layer_inputs, state)

      if len(y_list) > k:
        assert layer_inputs is not None and layer_inputs.shape[-2] == k
        x = y_list[-1]
        layer_inputs = jnp.concatenate(
            [layer_inputs, x[..., None, :]], axis=-2
        )

        M, state = compression(layer_inputs, state)

        layer_inputs = jnp.einsum('...id,...dik->...kd', layer_inputs, M)

      y_q = InputSelector(config=cfg, name=f'input_selector_q_{lyr}')(
          layer_inputs
      )
      y_kv = InputSelector(config=cfg, name=f'input_selector_kv_{lyr}')(
          layer_inputs
      )
      y_new = block(config=cfg, name=f'encoderdecoderblock_{lyr}')(
          y_q,
          y_kv,
          y_kv,
          decoder_mask,
          encoder_decoder_mask,
          deterministic,
      )
      y_list.append(y_new)

    if len(y_list) <= k:
      layer_inputs = jnp.stack(y_list, axis=-2)
    else:
      assert layer_inputs is not None and layer_inputs.shape[-2] == k
      x = y_list[-1]
      layer_inputs = jnp.concatenate([layer_inputs, x[..., None, :]], axis=-2)

      compression_final = MiddleSumCompression(
          config=cfg, k=k, name='compression_final'
      )

      M, _ = compression_final(layer_inputs, state)

      layer_inputs = jnp.einsum('...id,...dik->...kd', layer_inputs, M)

    y_out = InputSelector(config=cfg, name='input_selector_out')(layer_inputs)
    return y_out


class TransformerLM(nn.Module):
  """Transformer Language Model with exposure to Decoder."""

  docfg: DoConfig

  def setup(self):
    cfg = self.docfg
    self.embed = nn.Embed(
        num_embeddings=cfg.V,
        features=cfg.D,
        embedding_init=fsdp.init('embedding', cfg),
    )
    self.pos_embed = nn.Embed(
        num_embeddings=cfg.L,
        features=cfg.D,
        embedding_init=fsdp.init('embedding', cfg),
    )

    if not cfg.experimental_model:
      self.decoder = DefaultDecoder(cfg)
    elif (cfg.dca_mix_q or cfg.dca_mix_k or cfg.dca_mix_v
          or cfg.dca_mlp_read != 'none'):
      # Any per-axis mixing OR a non-"none" MLP read selects the post-proj
      # path. (depth_mem gates only on the mix flags and silently ignores
      # dca_mlp_read when no axis is mixed; we route it here so an
      # MLP-read-only config actually exercises the depth-pooled MLP.)
      self.decoder = DcaPostProjDecoder(cfg)
    else:
      self.decoder = DcaIncrementalDecoder(cfg)
    self.out_ln = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)

    if not cfg.logits_via_embedding:
      self.logit_dense = nn.Dense(
          cfg.V,
          dtype=cfg.dtype,
          kernel_init=fsdp.init('head', cfg),
          bias_init=cfg.bias_init,
          name='logit_dense',
      )

  def __call__(self, y_BxL, deterministic: Optional[bool] = None):
    cfg = self.docfg
    y_BxLxD = self.embed(y_BxL)
    y_BxLxD += self.pos_embed(jnp.arange(0, y_BxL.shape[1])[None, ...])

    # Causal mask for self-attention
    L = y_BxL.shape[1]
    mask = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))

    y_BxLxD = self.decoder(
        y_BxLxD, decoder_mask=mask, deterministic=deterministic
    )

    y_BxLxD = self.out_ln(y_BxLxD)

    if cfg.logits_via_embedding:
      logits_BxLxV = self.embed.attend(y_BxLxD.astype(jnp.float32))
      # Normalize logits by sqrt(D) to prevent overflow.
      logits_BxLxV = logits_BxLxV / jnp.sqrt(cfg.D)
    else:
      logits_BxLxV = self.logit_dense(y_BxLxD)

    return logits_BxLxV
