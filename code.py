
import dataclasses
from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp

from nanodo import fsdp
from nanodo import model as base_model


@dataclasses.dataclass
class DepthMemConfig(base_model.DoConfig):
  """DoConfig extended with depth-memory variant knobs."""

  variant: str = "baseline"  # one of {"baseline", "dca", "moda", "mhc"}

  # DCA: how to pick past-layer slots for the GRN reader.
  #   - "full"          : keep every prior layer (most general, biggest GRN).
  #   - "first_k"       : keep the earliest k slots only.
  #   - "last_k"        : keep the most recent k slots only.
  #   - "first_last_k"  : keep first-k + last-k (original DCA paper).
  dca_select_mode: str = "first_last_k"
  past_layers_k: int = 2  # used by first_k / last_k / first_last_k modes

  # DCA: per-axis cross-layer mixing switches. Three independent booleans
  # control whether each of Q, K, V mixes across past layers (post-projection,
  # per-head). The combination decides which DCA flavor runs:
  #
  #   - All three False (DEFAULT) → "hidden" DCA. GRN aggregates past
  #     hidden states in D-dim with no head structure (the original
  #     lucidrains design). `dca_select_mode` / `past_layers_k` apply.
  #     `dca_mlp_read` / `dca_mlp_write` are IGNORED in this mode.
  #   - Any True               → "post_proj" DCA. Per-head GRN aggregates
  #     past projections of the enabled axes. `dca_mlp_read` / `dca_mlp_write`
  #     apply; `dca_select_mode` / `past_layers_k` are IGNORED (always full).
  #
  # "All False" is a convention — it routes through hidden DCA, which is
  # not "no mixing" but rather "mixing in hidden space instead of QKV
  # space". Set at least one mix_* flag to opt into post-projection.
  dca_mix_q: bool = False
  dca_mix_k: bool = False
  dca_mix_v: bool = False

  # DCA MLP — two orthogonal axes (post_proj DCA only; hidden DCA ignores both).
  #
  # dca_mlp_read: how the MLP builds its INPUT from a depth-history GRN pool.
  #   `G(.)` is a 1-output GRNReader over [past stored slots | current slot];
  #   `x` is the block's residual stream, `a` the attention output. The current
  #   slot is always pooled (so it is gated), unlike the old design.
  #   - "none"             : pre_ff = norm(x + a)            (standard pre-FF)
  #   - "grn_pool"         : pre_ff = norm(G([hist, a]))     (residual unused)
  #   - "grn_pool_res_in"  : pre_ff = norm(G([hist, x + a])) (residual folded
  #                          into the current slot, hence into stored history)
  #   - "grn_pool_res_out" : pre_ff = norm(G([hist, a]) + x) (residual added
  #                          outside the pool, NOT stored to history)
  # dca_mlp_write: whether the MLP writes an extra depth K/V slot that later
  #   blocks' K/V aggregation can read (MoDA-style). Independent of dca_mlp_read.
  #   - "none"     : MLP writes nothing.
  #   - "moda_kv"  : MLP projects pre_ff to a K/V slot (reuses `MoDAKVProj`),
  #                  consumed by later blocks' k/v `_aggregate`.
  dca_mlp_read: str = "none"
  dca_mlp_write: str = "none"

  # When True, GRNReader sows post-ReLU gating weights and the absolute
  # source-layer index for every block. Required for the depth-attention
  # analysis pass (use `mutable=["intermediates"]` when calling `apply`).
  dca_capture_grn_weights: bool = False

  # DCA GRN bias placement (applies to BOTH the hidden `GRNReader` and the
  # post_proj `PerHeadGRNReader`):
  #   - False (DEFAULT): per-layer coeff = relu(gate) + bias. Bias is added
  #       AFTER the relu, so coefficients may be negative. Permits the
  #       split-einsum trick (part1 + part2) that never materializes the
  #       [O, K, B, L, D] aggregate.
  #   - True : per-layer coeff = relu(gate + bias). Bias sits INSIDE the relu
  #       (matches lucidrains / `dca_ref.py`'s InputSelector); coefficients are
  #       non-negative. relu couples gate and bias so the split is invalid —
  #       a [K, B, L, (H,) D] coefficient is materialized per output (looped
  #       over the small O axis to keep peak memory at ~mem, not O*mem).
  grn_bias_inside_relu: bool = False

  # DCA GRN gating activation (see `_GRN_ACTIVATIONS`). Elementwise options:
  # "relu" (default), "swish"/"silu", "gelu", "tanh", "sigmoid", "identity".
  # Axis-wise option: "softmax" normalizes the gate over the depth/layer axis
  # K, turning each token's gate into a convex combination over source layers.
  # Applies to both readers and both `grn_bias_inside_relu` placements.
  grn_act: str = "relu"

  # MoDA: full reference attention or chunk-visible depth retrieval.
  moda_impl: str = "naive"  # one of {"naive", "chunked"}
  moda_chunk_size: int = 64
  moda_include_mlp_kv: bool = True
  moda_depth_attn_mode: str = "causal"  # one of {"causal", "same_position"}

  # MoDA attention-dilution ablations (naive + causal only). Defaults reproduce
  # the plain joint-causal softmax. For a query at layer ell, time t:
  #   row  = current-layer causal self-attn; col = prior-layer slots at time==t;
  #   rest = prior-layer depth slots strictly before the query (time < t).
  moda_rest_select: str = "all"      # one of {"all", "topk", "global_topk",
                                     #   "entmax"}. "topk": protect row+col,
                                     #   hard top-k over rest. "global_topk":
                                     #   hard top-k over the whole [row|col|rest]
                                     #   (no protection; single-stream only) --
                                     #   the hard analog of "entmax".
  moda_rest_topk: int = 0            # absolute per-head k; required > 0 when
                                     #   rest_select in {topk, global_topk}. For
                                     #   "topk" it is k over rest; for
                                     #   "global_topk" it is k over the whole
                                     #   joint. |valid| < k keeps all valid.
  moda_entmax_alpha: float = 1.5     # alpha for "entmax" (2.0 == sparsemax).
  moda_two_stream: bool = False      # split into (row+col) and (rest) streams.
  moda_two_stream_gate_init: float = 0.0  # per-head gate logit init.
  moda_flash_block: int = 128  # key-tile size for moda_impl="flash" streaming.
                               #   Independent of moda_chunk_size (windowed impl).

  # mHC: stream count, Sinkhorn iterations, initial gate scale, and H_res mode.
  num_streams: int = 4
  sinkhorn_iters: int = 20
  mhc_init_gating_factor: float = 0.01
  mhc_h_res_mode: str = "sinkhorn"  # one of {"sinkhorn", "identity"}


def _cfg_dtype(cfg) -> jnp.dtype:
  return jnp.dtype(cfg.dtype)


def _assert_heads_divide_dim(cfg):
  assert cfg.D % cfg.H == 0, f"D {cfg.D} not divisible by H {cfg.H}"


def _dca_uses_post_proj(cfg) -> bool:
  """Whether the DCA path runs per-axis post-projection mixing.

  True iff any of `dca_mix_q`, `dca_mix_k`, `dca_mix_v` is enabled.
  When False, DCA falls through to the original lucidrains hidden-space
  GRN — i.e. "all False" routes through hidden mode, not "no mixing".
  """
  return cfg.dca_mix_q or cfg.dca_mix_k or cfg.dca_mix_v


def _dca_source_indices(valid_K: int, mode: str, k: int) -> tuple[int, ...]:
  """Static (Python-int) source-layer indices for a DCA selection.

  Slot semantics in the DCA memory buffer:
    - index 0           = token embedding (input to block 0)
    - index 1..valid_K-1 = output of block (index-1)

  `valid_K` is layer_index + 1 in the per-block reader, or N + 1 in the
  final pool. Returned tuple length is the effective K seen by the GRN.
  """
  if mode == "full":
    return tuple(range(valid_K))
  assert k > 0, f"past_layers_k must be > 0 for dca_select_mode={mode!r}"
  if mode == "first_k":
    return tuple(range(min(k, valid_K)))
  if mode == "last_k":
    n = min(k, valid_K)
    return tuple(range(valid_K - n, valid_K))
  if mode == "first_last_k":
    if valid_K <= 2 * k:
      return tuple(range(valid_K))
    return tuple(list(range(k)) + list(range(valid_K - k, valid_K)))
  raise NotImplementedError(f"unknown dca_select_mode: {mode!r}")


def _dca_select(
    mem_KxBxLxD: jax.Array, mode: str, k: int
) -> tuple[jax.Array, tuple[int, ...]]:
  """Pick past-layer slots from a DCA memory buffer.

  Returns `(selected_mem, source_indices)` — the sliced memory and the
  absolute source-layer indices it corresponds to. `_dca_source_indices`
  remains the single source of truth for which slots are visible; this
  function just performs the data movement, grouping consecutive indices
  into contiguous runs so each selection is a slice (or a concat of slices)
  rather than a general gather.

  `_dca_source_indices` is kept as a separate helper for the one call
  site that needs indices without a memory buffer (model `setup()`, where
  the GRN reader for the final pool is constructed before any forward
  pass has produced the memory).
  """
  K = mem_KxBxLxD.shape[0]
  source_indices = _dca_source_indices(K, mode, k)
  assert all(0 <= i < K for i in source_indices), (source_indices, K)

  if source_indices == tuple(range(K)):
    return mem_KxBxLxD, source_indices

  runs = []
  start = prev = source_indices[0]
  for idx in source_indices[1:]:
    if idx == prev + 1:
      prev = idx
      continue
    runs.append((start, prev + 1))
    start = prev = idx
  runs.append((start, prev + 1))

  if len(runs) == 1:
    start, end = runs[0]
    return mem_KxBxLxD[start:end], source_indices
  return (
      jnp.concatenate(
          [mem_KxBxLxD[start:end] for start, end in runs], axis=0
      ),
      source_indices,
  )


def _DEAD_VALUE(dtype) -> jax.Array:
  """0-d sentinel returned by variants whose outer loop ignores the
  per-block 'hidden state' slot (DCA reads memory directly, mHC carries
  state in `streams`). Returning a real jax array keeps the tuple
  well-typed for `nn.remat` (which rejects `None` in its outputs).
  """
  return jnp.zeros((), dtype=dtype)


# GRN gate activations selectable via `cfg.grn_act`. Elementwise entries act
# pointwise on the gate; "softmax" normalizes over the depth/layer axis K
# (axis 0) — every reader call site has K as axis 0, so this is consistent for
# the hidden and per-head readers and for both bias placements.
_GRN_ACTIVATIONS = {
    "relu": jax.nn.relu,
    "swish": jax.nn.swish,  # jax aliases swish == silu
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "identity": lambda x: x,
    "softmax": partial(jax.nn.softmax, axis=0),
}


def _grn_activation(name: str):
  """Resolve `cfg.grn_act` to the callable applied to the GRN gate."""
  try:
    return _GRN_ACTIVATIONS[name]
  except KeyError as e:
    raise NotImplementedError(
        f"unknown grn_act {name!r}; choose from {sorted(_GRN_ACTIVATIONS)}"
    ) from e


class _GRNReaderBase(nn.Module):
  """Shared bookkeeping for GRN readers with different projection layouts."""

  cfg: DepthMemConfig
  num_layers_visible: int
  num_outputs: int
  source_layer_indices: tuple[int, ...] = ()

  def _checked_k_and_outputs(self, mem) -> tuple[int, int]:
    K = mem.shape[0]
    assert K == self.num_layers_visible, (K, self.num_layers_visible)
    return K, self.num_outputs

  def _grn_bias(self, shape: tuple[int, ...]) -> jax.Array:
    return self.param(
        "grn_bias",
        nn.initializers.ones,
        shape,
        _cfg_dtype(self.cfg),
    )

  def _maybe_sow_grn_analysis(self, weights: jax.Array):
    if not self.cfg.dca_capture_grn_weights:
      return
    self.sow("intermediates", "grn_weights", weights)
    if self.source_layer_indices:
      self.sow(
          "intermediates",
          "source_layer_indices",
          jnp.asarray(self.source_layer_indices, dtype=jnp.int32),
      )

  def _outputs_tuple(self, out_OxBxLxAny: jax.Array):
    return tuple(out_OxBxLxAny[i] for i in range(self.num_outputs))


class GRNReader(_GRNReaderBase):
  """N-output GRN-v3 reader over a depth stack of hidden states.

  GRN-v3 dataflow (default, `grn_bias_inside_relu=False`):
    gate[k,b,l,o]        = Dense(RMSNorm(mem))[k,b,l,o]
    aggregate[o,k,b,l,d] = act(gate[k,b,l,o]) + bias[o,k,d]
    out[o,b,l,d]         = sum_k mem[k,b,l,d] * aggregate[o,k,b,l,d]

  where `act = cfg.grn_act` (default "relu"; see `_GRN_ACTIVATIONS`).

  With `grn_bias_inside_relu=True` the bias moves inside the activation to
  match lucidrains / `dca_ref.py`'s InputSelector exactly:
    aggregate[o,k,b,l,d] = act(gate[k,b,l,o] + bias[o,k,d])
  For relu, act(a+b) != act(a)+b, so this is a genuinely different function
  (coefficients become non-negative) — see the config docstring.

  Implementation detail: the formula above is distributed into two einsums

      part1[o,b,l,d] = sum_k mem[k,b,l,d] * weights[k,b,l,o]   (token gating)
      part2[o,b,l,d] = sum_k mem[k,b,l,d] * bias[o,k,d]        (per-layer bias mix)
      out = part1 + part2

  which is mathematically equivalent (linearity of `sum_k` over `+`) but
  avoids materializing the [O, K, B, L, D] aggregate tensor — that
  intermediate dominated DCA peak memory at training-realistic shapes
  (e.g. ~13 GB at O=3, K=4, B=256, L=2048, D=1024, bf16). The two einsums
  fuse cleanly on TPU and total FLOPs are unchanged.
  """

  @nn.compact
  def __call__(self, mem_KxBxLxD: jax.Array):
    cfg = self.cfg
    K, O = self._checked_k_and_outputs(mem_KxBxLxD)

    bias_OxKxD = self._grn_bias((O, K, cfg.D))

    normed = nn.RMSNorm(dtype=cfg.dtype, name="grn_norm")(mem_KxBxLxD)
    gate_KxBxLxO = nn.Dense(
        features=O,
        use_bias=False,
        kernel_init=nn.initializers.normal(stddev=1e-2),
        dtype=cfg.dtype,
        name="grn_proj",
    )(normed)

    # Analysis hook: sow the gating weights and the absolute source-layer
    # indices. The sown values land in
    # `intermediates["...path.../GRNReader_0/grn_weights"]` and can be
    # collected by calling `model.apply(..., mutable=["intermediates"])`.
    act = _grn_activation(cfg.grn_act)
    if cfg.grn_bias_inside_relu:
      # coeff[o,k,b,l,d] = act(gate + bias): the activation couples gate and
      # bias, so the distributive split below is invalid. Materialize one
      # [K,B,L,D] coefficient per output, looping over the small O axis to keep
      # peak memory at ~mem (not O*mem). Sow the raw pre-bias gate for analysis.
      self._maybe_sow_grn_analysis(gate_KxBxLxO)
      outs = []
      for o in range(O):
        coeff_KxBxLxD = act(
            gate_KxBxLxO[..., o, None] + bias_OxKxD[o].reshape(K, 1, 1, cfg.D)
        )
        outs.append(jnp.einsum("kbld,kbld->bld", mem_KxBxLxD, coeff_KxBxLxD))
      return tuple(outs)

    weights_KxBxLxO = act(gate_KxBxLxO)
    self._maybe_sow_grn_analysis(weights_KxBxLxO)

    # coeff = act(gate) + bias, distributed into two einsums (linearity of
    # sum_k over +) so the [O,K,B,L,D] aggregate is never materialized.
    # part1[o,b,l,d] = sum_k mem[k,b,l,d] * weights[k,b,l,o]
    part1_OxBxLxD = jnp.einsum(
        "kbld,kblo->obld", mem_KxBxLxD, weights_KxBxLxO
    )
    # part2[o,b,l,d] = sum_k mem[k,b,l,d] * bias[o,k,d]
    part2_OxBxLxD = jnp.einsum("kbld,okd->obld", mem_KxBxLxD, bias_OxKxD)
    out_OxBxLxD = part1_OxBxLxD + part2_OxBxLxD
    return self._outputs_tuple(out_OxBxLxD)


class PerHeadGRNReader(_GRNReaderBase):
  """Per-head GRN-v3 reader over a depth stack of [K, B, L, H, Dh] tensors.

  Mirrors `GRNReader` but each head has its own gating function and bias
  (`act = cfg.grn_act`, default "relu"):

      weights[k,b,l,h,o] = act(Linear_per_head(RMSNorm(mem)))[k,b,l,h,o]
      out[o,b,l,h,dh]    = sum_k mem[k,b,l,h,dh] * (weights[k,b,l,h,o] + bias[o,k,h,dh])

  Used by the `kv_post_proj` DCA aggregation to let each attention head route
  across past layers independently. Same split-einsum trick as `GRNReader` is
  applied to avoid materializing the [O, K, B, L, H, Dh] aggregate tensor.

  The projection kernel is `[H, Dh, num_outputs]` — per-head, not shared
  across heads — so the model can learn head-specific routing functions
  (not just head-specific bias).
  """

  @nn.compact
  def __call__(self, mem_KxBxLxHxDh: jax.Array):
    cfg = self.cfg
    K, O = self._checked_k_and_outputs(mem_KxBxLxHxDh)
    H = mem_KxBxLxHxDh.shape[-2]
    Dh = mem_KxBxLxHxDh.shape[-1]

    bias_OxKxHxDh = self._grn_bias((O, K, H, Dh))

    # Per-head RMSNorm over the Dh axis (default nn.RMSNorm normalizes over
    # the last axis; scale is shape [Dh] shared across heads — adequate for
    # this first cut; per-head scale is a low-risk follow-up).
    normed_KxBxLxHxDh = nn.RMSNorm(
        dtype=cfg.dtype, name="grn_norm"
    )(mem_KxBxLxHxDh)

    # Per-head projection: kernel [H, Dh, O] is independent across heads.
    proj_kernel_HxDhxO = self.param(
        "grn_proj_kernel",
        nn.initializers.normal(stddev=1e-2),
        (H, Dh, O),
        _cfg_dtype(cfg),
    )
    gate_KxBxLxHxO = jnp.einsum(
        "kblhd,hdo->kblho", normed_KxBxLxHxDh, proj_kernel_HxDhxO
    )

    act = _grn_activation(cfg.grn_act)
    if cfg.grn_bias_inside_relu:
      # coeff = act(gate + bias); see GRNReader for why the split is invalid.
      # Loop over O, materializing one [K,B,L,H,Dh] coefficient at a time.
      self._maybe_sow_grn_analysis(gate_KxBxLxHxO)
      outs = []
      for o in range(O):
        coeff_KxBxLxHxDh = act(
            gate_KxBxLxHxO[..., o, None]
            + bias_OxKxHxDh[o].reshape(K, 1, 1, H, Dh)
        )
        outs.append(
            jnp.einsum("kblhd,kblhd->blhd", mem_KxBxLxHxDh, coeff_KxBxLxHxDh)
        )
      return tuple(outs)

    weights_KxBxLxHxO = act(gate_KxBxLxHxO)
    self._maybe_sow_grn_analysis(weights_KxBxLxHxO)

    # Distributive split (matches GRNReader's trick):
    # part1[o,b,l,h,dh] = sum_k mem[k,b,l,h,dh] * weights[k,b,l,h,o]
    # part2[o,b,l,h,dh] = sum_k mem[k,b,l,h,dh] * bias[o,k,h,dh]
    part1_OxBxLxHxDh = jnp.einsum(
        "kblhd,kblho->oblhd", mem_KxBxLxHxDh, weights_KxBxLxHxO
    )
    part2_OxBxLxHxDh = jnp.einsum(
        "kblhd,okhd->oblhd", mem_KxBxLxHxDh, bias_OxKxHxDh
    )
    out_OxBxLxHxDh = part1_OxBxLxHxDh + part2_OxBxLxHxDh
    return self._outputs_tuple(out_OxBxLxHxDh)


def sinkhorn_knopp(
    logits: jax.Array, num_iter: int, eps: float = 1e-6
) -> jax.Array:
  """Differentiable Sinkhorn-Knopp normalization in float32."""
  orig_dtype = logits.dtype
  logits_f32 = logits.astype(jnp.float32)
  m0 = jnp.exp(logits_f32 - logits_f32.max(axis=-1, keepdims=True))

  def body(_, m):
    m = m / jnp.clip(m.sum(axis=-1, keepdims=True), min=eps)
    m = m / jnp.clip(m.sum(axis=-2, keepdims=True), min=eps)
    return m

  m = jax.lax.fori_loop(0, num_iter, body, m0)
  return m.astype(orig_dtype)


class MHCMappings(nn.Module):
  """Compute (H_pre, H_post, H_res) per token for mHC."""

  cfg: DepthMemConfig

  @nn.compact
  def __call__(self, streams_BxLxnxD: jax.Array):
    cfg = self.cfg
    n = cfg.num_streams
    B, L, _, D = streams_BxLxnxD.shape
    flat_BxLxnD = streams_BxLxnxD.reshape(B, L, n * D)

    use_sinkhorn = cfg.mhc_h_res_mode == "sinkhorn"
    use_identity = cfg.mhc_h_res_mode == "identity"
    assert use_sinkhorn or use_identity, (
        f"unknown mhc_h_res_mode {cfg.mhc_h_res_mode!r}"
    )

    proj_dim = 2 * n + (n * n if use_sinkhorn else 0)
    proj = nn.Dense(
        features=proj_dim,
        use_bias=False,
        kernel_init=nn.initializers.xavier_uniform(),
        dtype=cfg.dtype,
        name="mapping_proj",
    )(flat_BxLxnD)

    rms = jnp.sqrt(
        jnp.mean(flat_BxLxnD.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
    )
    r = (1.0 / (rms + 1e-6)).astype(cfg.dtype)

    init_a = cfg.mhc_init_gating_factor
    alpha_pre = self.param(
        "alpha_pre", lambda _: jnp.full((1,), init_a, _cfg_dtype(cfg))
    )
    alpha_post = self.param(
        "alpha_post", lambda _: jnp.full((1,), init_a, _cfg_dtype(cfg))
    )
    alpha_parts = [
        jnp.broadcast_to(alpha_pre, (n,)),
        jnp.broadcast_to(alpha_post, (n,)),
    ]
    if use_sinkhorn:
      alpha_res = self.param(
          "alpha_res", lambda _: jnp.full((1,), init_a, _cfg_dtype(cfg))
      )
      alpha_parts.append(jnp.broadcast_to(alpha_res, (n * n,)))

    bias_proj = self.param(
        "mapping_bias", nn.initializers.zeros, (proj_dim,), _cfg_dtype(cfg)
    )

    alpha = jnp.concatenate(alpha_parts, axis=-1).reshape(1, 1, proj_dim)
    h = r * proj * alpha + bias_proj.reshape(1, 1, proj_dim)

    h_pre = jax.nn.sigmoid(h[..., :n])
    h_post = 2.0 * jax.nn.sigmoid(h[..., n : 2 * n])
    if use_sinkhorn:
      h_res_logits = h[..., 2 * n :].reshape(B, L, n, n)
      h_res = sinkhorn_knopp(h_res_logits, num_iter=cfg.sinkhorn_iters)
    else:
      eye = jnp.eye(n, dtype=_cfg_dtype(cfg))
      h_res = jnp.broadcast_to(eye, (B, L, n, n))
    return h_pre, h_post, h_res.astype(cfg.dtype)


def _mhc_aggregate(
    streams_BxLxnxD: jax.Array, h_pre_BxLxn: jax.Array
) -> jax.Array:
  return jnp.einsum("blnd,bln->bld", streams_BxLxnxD, h_pre_BxLxn)


def _mhc_step(
    streams_BxLxnxD: jax.Array,
    h_post_BxLxn: jax.Array,
    h_res_BxLxnxn: jax.Array,
    f_out_BxLxD: jax.Array,
    h_res_is_identity: bool = False,
) -> jax.Array:
  expand_BxLxnxD = jnp.einsum("bln,bld->blnd", h_post_BxLxn, f_out_BxLxD)
  if h_res_is_identity:
    return streams_BxLxnxD + expand_BxLxnxD
  residual_BxLxnxD = jnp.einsum(
      "blij,bljd->blid", h_res_BxLxnxn, streams_BxLxnxD
  )
  return residual_BxLxnxD + expand_BxLxnxD


class MoDAKVProj(nn.Module):
  """Project hidden states to one K/V depth slot."""

  cfg: DepthMemConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg
    _assert_heads_divide_dim(cfg)
    Dh = cfg.D // cfg.H
    linear = partial(
        nn.DenseGeneral,
        axis=-1,
        features=(cfg.H, Dh),
        kernel_init=fsdp.init("attn_in_proj", cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )
    return linear(name="key")(x_BxLxD), linear(name="value")(x_BxLxD)


def _flatten_depth_kv(depth_k, depth_v):
  """[P,S,B,L,H,Dh] -> [B,P*S*L,H,Dh]."""
  P, S, B, L, H, Dh = depth_k.shape
  flat_k = jnp.transpose(depth_k, (2, 0, 1, 3, 4, 5)).reshape(
      B, P * S * L, H, Dh
  )
  flat_v = jnp.transpose(depth_v, (2, 0, 1, 3, 4, 5)).reshape(
      B, P * S * L, H, Dh
  )
  return flat_k, flat_v


def _combined_softmax_attention(q, k_all, v_all, mask_LxK, dtype):
  logits = jnp.einsum("...qhd,...khd->...hqk", q, k_all).astype(jnp.float32)
  logits = jnp.where(
      mask_LxK[None, None, :, :], logits, jnp.finfo(jnp.float32).min
  )
  weights = jax.nn.softmax(logits, axis=-1).astype(dtype)
  return jnp.einsum("...hqk,...khd->...qhd", weights, v_all)


def _moda_attention_same_position(
    q_BxLxHxDh,
    k_cur,
    v_cur,
    depth_k,
    depth_v,
    dtype,
):
  """MoDA path where each query reads depth KV only at the same position."""
  B, L, H, Dh = q_BxLxHxDh.shape
  P, S = depth_k.shape[:2]

  seq_logits = jnp.einsum(
      "bqhd,bkhd->bhqk", q_BxLxHxDh, k_cur
  ).astype(jnp.float32)
  seq_mask = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))
  seq_logits = jnp.where(
      seq_mask, seq_logits, jnp.finfo(jnp.float32).min
  )

  if P == 0:
    weights = jax.nn.softmax(seq_logits, axis=-1).astype(dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v_cur)

  depth_len = P * S
  # Per-position depth logits: query l reads only its own (P*S) depth slots.
  # Fold the [P,S,B,L,...] -> [B,L,M,...] permutation straight into the
  # contraction so XLA never materializes a transposed copy of the (large)
  # depth tensors -- the (p,s) axes ride along as extra output dims and the
  # trailing reshape that merges them into M is a free, contiguous view.
  depth_logits = (
      jnp.einsum("blhd,psblhd->bhlps", q_BxLxHxDh, depth_k)
      .astype(jnp.float32)
      .reshape(B, H, L, depth_len)
  )

  # Joint softmax over the seq (L) and per-position depth (P*S) logits without
  # materializing the concatenated [B,H,L,L+P*S] tensor: subtract the shared
  # row max across both sources, then normalize by the shared denominator. This
  # is numerically identical to a softmax over the concatenation but keeps the
  # two logit tensors separate, trimming the float32 softmax peak by one copy.
  joint_max = jax.lax.stop_gradient(
      jnp.maximum(
          seq_logits.max(axis=-1, keepdims=True),
          depth_logits.max(axis=-1, keepdims=True),
      )
  )
  seq_exp = jnp.exp(seq_logits - joint_max)
  depth_exp = jnp.exp(depth_logits - joint_max)
  denom = seq_exp.sum(axis=-1, keepdims=True) + depth_exp.sum(
      axis=-1, keepdims=True
  )
  seq_weights = (seq_exp / denom).astype(dtype)
  depth_weights = (depth_exp / denom).reshape(B, H, L, P, S).astype(dtype)

  seq_out = jnp.einsum("bhlk,bkhd->blhd", seq_weights, v_cur)
  depth_out = jnp.einsum("bhlps,psblhd->blhd", depth_weights, depth_v)
  return seq_out + depth_out


def _moda_attention_for_queries(
    q_slice,
    q_positions,
    k_cur,
    v_cur,
    depth_k,
    depth_v,
    depth_positions,
    depth_attn_mode: str,
    dtype,
):
  """Attend one query slice to full sequence KV and visible depth KV."""
  key_positions = jnp.arange(k_cur.shape[1])
  seq_mask = q_positions[:, None] >= key_positions[None, :]
  if depth_k.shape[0] == 0:
    return _combined_softmax_attention(q_slice, k_cur, v_cur, seq_mask, dtype)

  past_k, past_v = _flatten_depth_kv(depth_k, depth_v)
  k_all = jnp.concatenate([k_cur, past_k], axis=1)
  v_all = jnp.concatenate([v_cur, past_v], axis=1)
  if depth_attn_mode == "causal":
    depth_mask = q_positions[:, None] >= depth_positions[None, :]
  elif depth_attn_mode == "same_position":
    depth_mask = q_positions[:, None] == depth_positions[None, :]
  else:
    raise NotImplementedError(
        f"unknown moda_depth_attn_mode {depth_attn_mode!r}"
    )
  mask = jnp.concatenate([seq_mask, depth_mask], axis=-1)
  return _combined_softmax_attention(q_slice, k_all, v_all, mask, dtype)


def _moda_attention_chunked(
    q_BxLxHxDh,
    k_cur,
    v_cur,
    depth_k,
    depth_v,
    chunk_size: int,
    depth_attn_mode: str,
    dtype,
):
  """Full sequence attention plus chunk-visible depth attention.

  When the sequence length is divisible by `chunk_size`, dispatches to a
  vmap-based path whose HLO size is independent of `L / chunk_size` — useful
  for long-sequence training where a Python-unrolled loop would explode
  compile time and TPU pipelining. Falls back to the pyloop path when
  divisibility doesn't hold.
  """
  _, L, _, _ = q_BxLxHxDh.shape
  if depth_attn_mode == "same_position":
    return _moda_attention_same_position(
        q_BxLxHxDh, k_cur, v_cur, depth_k, depth_v, dtype,
    )
  C = min(max(int(chunk_size), 1), L)
  if L > 0 and L % C == 0:
    return _moda_attention_chunked_vmap(
        q_BxLxHxDh, k_cur, v_cur, depth_k, depth_v, C, depth_attn_mode, dtype,
    )
  return _moda_attention_chunked_pyloop(
      q_BxLxHxDh, k_cur, v_cur, depth_k, depth_v, C, depth_attn_mode, dtype,
  )


def _moda_attention_chunked_pyloop(
    q_BxLxHxDh,
    k_cur,
    v_cur,
    depth_k,
    depth_v,
    C: int,
    depth_attn_mode: str,
    dtype,
):
  """Python-unrolled chunk loop. Fallback when L % C != 0."""
  _, L, _, _ = q_BxLxHxDh.shape
  P, S = depth_k.shape[:2]
  chunks = []
  for start in range(0, L, C):
    end = min(start + C, L)
    q_chunk = q_BxLxHxDh[:, start:end]
    q_positions = jnp.arange(start, end)
    depth_k_chunk = depth_k[:, :, :, start:end]
    depth_v_chunk = depth_v[:, :, :, start:end]
    depth_positions = jnp.tile(q_positions[None, :], (P * S, 1)).reshape(-1)
    chunks.append(
        _moda_attention_for_queries(
            q_chunk,
            q_positions,
            k_cur,
            v_cur,
            depth_k_chunk,
            depth_v_chunk,
            depth_positions,
            depth_attn_mode,
            dtype,
        )
    )
  return jnp.concatenate(chunks, axis=1)


def _moda_attention_chunked_vmap(
    q_BxLxHxDh,
    k_cur,
    v_cur,
    depth_k,
    depth_v,
    C: int,
    depth_attn_mode: str,
    dtype,
):
  """vmap path. Requires L % C == 0. The HLO emitted by JAX is invariant
  in `L / C`, which keeps compile time and on-device work scaling cleanly
  for long sequences.

  The depth-KV tensor `[P, S, B, L, H, Dh]` is reshaped so the
  `(num_chunks, C)` axes replace the seq axis, then transposed to put
  `num_chunks` first so it can be the vmap axis. `k_cur` / `v_cur` are
  closed over (`in_axes=None`) since every chunk attends to the same
  full-sequence KV.
  """
  B, L, H, Dh = q_BxLxHxDh.shape
  P, S = depth_k.shape[:2]
  num_chunks = L // C

  q_NxBxCxHxDh = (
      q_BxLxHxDh.reshape(B, num_chunks, C, H, Dh).transpose(1, 0, 2, 3, 4)
  )
  # [P, S, B, L, H, Dh] -> [P, S, B, num_chunks, C, H, Dh]
  #                    -> [num_chunks, P, S, B, C, H, Dh]
  dk_NxPxSxBxCxHxDh = (
      depth_k.reshape(P, S, B, num_chunks, C, H, Dh)
      .transpose(3, 0, 1, 2, 4, 5, 6)
  )
  dv_NxPxSxBxCxHxDh = (
      depth_v.reshape(P, S, B, num_chunks, C, H, Dh)
      .transpose(3, 0, 1, 2, 4, 5, 6)
  )
  chunk_starts = jnp.arange(num_chunks) * C  # [num_chunks]
  arange_C = jnp.arange(C)

  def per_chunk(q_chunk, dk_chunk, dv_chunk, start):
    q_positions = start + arange_C
    depth_positions = jnp.tile(q_positions[None, :], (P * S, 1)).reshape(-1)
    return _moda_attention_for_queries(
        q_chunk,
        q_positions,
        k_cur,
        v_cur,
        dk_chunk,
        dv_chunk,
        depth_positions,
        depth_attn_mode,
        dtype,
    )

  # vmap over the leading chunk axis. k_cur/v_cur are not mapped (they're
  # closed over the lexical scope and broadcast to every chunk).
  out_NxBxCxHxDh = jax.vmap(per_chunk, in_axes=(0, 0, 0, 0))(
      q_NxBxCxHxDh, dk_NxPxSxBxCxHxDh, dv_NxPxSxBxCxHxDh, chunk_starts,
  )
  # [num_chunks, B, C, H, Dh] -> [B, num_chunks, C, H, Dh] -> [B, L, H, Dh]
  return out_NxBxCxHxDh.transpose(1, 0, 2, 3, 4).reshape(B, L, H, Dh)


def _entmax_forward(z, alpha, axis, n_iter):
  """alpha-entmax via threshold bisection (Peters & Martins, 2019).

  Returns p with p_i = [(alpha-1) z_i - tau]_+^{1/(alpha-1)}, tau chosen so the
  weights sum to 1. alpha->1 recovers softmax; alpha==2 is sparsemax. Computed
  in float32; masked logits (very negative) map to exactly 0.
  """
  am1 = alpha - 1.0
  zs = z.astype(jnp.float32) * am1
  d = zs.shape[axis]
  max_val = jnp.max(zs, axis=axis, keepdims=True)
  tau_lo = max_val - 1.0                               # f(tau_lo) >= 0
  tau_hi = max_val - (1.0 / d) ** am1                  # f(tau_hi) <= 0
  dm = tau_hi - tau_lo
  p = jnp.maximum(zs - tau_lo, 0.0) ** (1.0 / am1)
  for _ in range(n_iter):
    dm = dm * 0.5
    tau_m = tau_lo + dm
    p = jnp.maximum(zs - tau_m, 0.0) ** (1.0 / am1)
    f_m = jnp.sum(p, axis=axis, keepdims=True) - 1.0
    tau_lo = jnp.where(f_m >= 0.0, tau_m, tau_lo)
  return p / jnp.sum(p, axis=axis, keepdims=True)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def _entmax(z, alpha=1.5, axis=-1, n_iter=24):
  return _entmax_forward(z, alpha, axis, n_iter)


@_entmax.defjvp
def _entmax_jvp(alpha, axis, n_iter, primals, tangents):
  (z,), (dz,) = primals, tangents
  p = _entmax_forward(z, alpha, axis, n_iter)
  # JVP of alpha-entmax: dp = s * (dz - <s,dz>/<s,1>), s_i = p_i^{2-alpha} on
  # the support (0 elsewhere). Linear in dz, so JAX transposes it for grad.
  s = jnp.where(p > 0.0, p ** (2.0 - alpha), 0.0)
  dz = dz.astype(p.dtype)
  num = jnp.sum(s * dz, axis=axis, keepdims=True)
  den = jnp.sum(s, axis=axis, keepdims=True)
  dp = s * (dz - num / den)
  return p, dp


def _topk_keep(logits_BHLK, valid_mask_LK, k):
  """Per-(B,H,query) keep-mask: True for the top-k entries (by logit) among the
  valid (causally-allowed) keys, False elsewhere. k>0 assumed; clamps to the
  key count. Used both for top-k over `rest` only and for global top-k over the
  whole [row|col|rest] joint."""
  neg = jnp.finfo(jnp.float32).min
  masked = jnp.where(valid_mask_LK[None, None], logits_BHLK, neg)
  K = logits_BHLK.shape[-1]
  k = min(int(k), K)
  thresh = jax.lax.top_k(masked, k)[0][..., -1:]        # kth-largest per row
  # AND valid_mask so NEG>=NEG ties (when |valid|<k) cannot leak masked keys.
  return (masked >= thresh) & valid_mask_LK[None, None]


def _row_only(row_logits, row_mask, v_cur, cfg, dtype):
  """P==0 (no depth): normalize over the causal row alone."""
  neg = jnp.finfo(jnp.float32).min
  row_l = jnp.where(row_mask[None, None], row_logits, neg)
  if (not cfg.moda_two_stream) and cfg.moda_rest_select == "entmax":
    w = _entmax(row_l, cfg.moda_entmax_alpha, axis=-1).astype(dtype)
  else:
    w = jax.nn.softmax(row_l, axis=-1).astype(dtype)
  return jnp.einsum("bhlk,bkhd->blhd", w, v_cur)


def _single_stream_out(row_logits, depth_logits, row_mask, col_mask, rest_mask,
                       v_cur, dv, cfg):
  """Joint normalization over [row | col | rest]. Reached only with
  rest_select in {topk, entmax} (all + single-stream stays on the old path)."""
  dtype = cfg.dtype
  neg = jnp.finfo(jnp.float32).min
  if cfg.moda_rest_select == "topk":
    depth_keep = (_topk_keep(depth_logits, rest_mask, cfg.moda_rest_topk)
                  | col_mask[None, None])             # protect row+col
  else:  # entmax / global_topk: all causal depth visible; selection is global
    depth_keep = (col_mask | rest_mask)[None, None]
  row_l = jnp.where(row_mask[None, None], row_logits, neg)
  depth_l = jnp.where(depth_keep, depth_logits, neg)
  joint = jnp.concatenate([row_l, depth_l], axis=-1)
  if cfg.moda_rest_select == "global_topk":
    # Hard analog of global entmax: top-k over the whole causal joint, no
    # row/col protection.
    valid = jnp.concatenate(
        [row_mask, col_mask | rest_mask], axis=-1)      # [L, L+M]
    keep = _topk_keep(joint, valid, cfg.moda_rest_topk)
    joint = jnp.where(keep, joint, neg)
  if cfg.moda_rest_select == "entmax":
    w = _entmax(joint, cfg.moda_entmax_alpha, axis=-1).astype(dtype)
  else:
    w = jax.nn.softmax(joint, axis=-1).astype(dtype)
  L = row_logits.shape[-1]
  return (jnp.einsum("bhlk,bkhd->blhd", w[..., :L], v_cur)
          + jnp.einsum("bhlm,bmhd->blhd", w[..., L:], dv))


def _two_stream_out(q, k_cur, v_cur, depth_k, depth_v, depth_logits, rest_mask,
                    dv, cfg, gate):
  """Two separately-normalized streams combined by a per-head gate.

  A = softmax over [row | col] (== same_position). B = rest only, normalized by
  rest_select (all=softmax / topk=hard-select+softmax / entmax). Rows with no
  rest contribute B=0 and are forced to A. out = sigmoid(g)*A + (1-sigmoid(g))*B.
  """
  dtype = cfg.dtype
  neg = jnp.finfo(jnp.float32).min
  A = _moda_attention_same_position(q, k_cur, v_cur, depth_k, depth_v, dtype)
  if cfg.moda_rest_select == "topk":
    keep = _topk_keep(depth_logits, rest_mask, cfg.moda_rest_topk)
  else:                                                  # all / entmax
    keep = rest_mask[None, None]
  rest_l = jnp.where(keep, depth_logits, neg)
  if cfg.moda_rest_select == "entmax":
    wB = _entmax(rest_l, cfg.moda_entmax_alpha, axis=-1)
  else:
    wB = jax.nn.softmax(rest_l, axis=-1)
  has_rest = jnp.any(rest_mask, axis=-1)                 # [L]
  wB = jnp.where(has_rest[None, None, :, None], wB, 0.0).astype(dtype)
  Bout = jnp.einsum("bhlm,bmhd->blhd", wB, dv)
  g = jax.nn.sigmoid(gate.astype(dtype))[None, None, :, None]   # [1,1,H,1]
  g = jnp.where(has_rest[None, :, None, None], g, jnp.ones((), dtype))
  return g * A + (1.0 - g) * Bout


def _moda_attention_naive_select(q, k_cur, v_cur, depth_k, depth_v, cfg, gate):
  """Naive (full-sequence) MoDA attention with the row/col/rest partition.

  row  = current-layer causal self-attn; col = prior-layer depth at time==t;
  rest = prior-layer depth at time<t. Dispatches on cfg.moda_two_stream /
  cfg.moda_rest_select. `gate` is the per-head [H] logit (two-stream only).
  """
  dtype = cfg.dtype
  _, L, _, _ = q.shape
  P, S = depth_k.shape[:2]
  t = jnp.arange(L)
  row_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k_cur).astype(jnp.float32)
  row_mask = t[:, None] >= t[None, :]                  # [L,L] causal
  if P == 0:
    return _row_only(row_logits, row_mask, v_cur, cfg, dtype)
  dk, dv = _flatten_depth_kv(depth_k, depth_v)          # [B,M,H,Dh]
  depth_time = jnp.tile(t[None, :], (P * S, 1)).reshape(-1)        # [M]
  depth_logits = jnp.einsum("bqhd,bmhd->bhqm", q, dk).astype(jnp.float32)
  col_mask = depth_time[None, :] == t[:, None]          # [L,M] same time
  rest_mask = depth_time[None, :] < t[:, None]          # [L,M] strictly past
  if cfg.moda_two_stream:
    return _two_stream_out(q, k_cur, v_cur, depth_k, depth_v,
                           depth_logits, rest_mask, dv, cfg, gate)
  return _single_stream_out(row_logits, depth_logits, row_mask, col_mask,
                            rest_mask, v_cur, dv, cfg)


def _tile_keys(keys_BNHD, vals_BNHD, kp_N, block):
  """Pad the key axis to a multiple of `block` and reshape into tiles with the
  tile axis leading (for lax.scan). Returns keys/vals [nt,B,block,H,Dh], key
  positions [nt,block], and a validity mask [nt,block] (False for padding).

  `vals_BNHD` may be None (keys-only tiling, e.g. the top-k buffer needs no
  values); in that case the returned vals is None and no extra tensor is
  allocated."""
  B, N, H, Dh = keys_BNHD.shape
  nt = -(-N // block)                       # ceil(N / block)
  n = nt * block
  pad = n - N
  keys = jnp.pad(keys_BNHD, ((0, 0), (0, pad), (0, 0), (0, 0)))
  keys = keys.reshape(B, nt, block, H, Dh).transpose(1, 0, 2, 3, 4)
  kp = jnp.pad(kp_N, (0, pad))
  kvalid = jnp.arange(n) < N                # [n]
  vals = None
  if vals_BNHD is not None:
    vals = jnp.pad(vals_BNHD, ((0, 0), (0, pad), (0, 0), (0, 0)))
    vals = vals.reshape(B, nt, block, H, Dh).transpose(1, 0, 2, 3, 4)
  return keys, vals, kp.reshape(nt, block), kvalid.reshape(nt, block)


def _flash_softmax_stream(q_BLHD, keys_BNHD, vals_BNHD, kp_N, q_pos_L, compare,
                          block, threshold=None):
  """Online-softmax over one key set, streamed over key-tiles (remat'd).

  compare: "causal" (q_pos >= key_pos) or "strict" (q_pos > key_pos).
  threshold: optional [B,H,L,1] — also require logit >= threshold (top-k pass 2).
  Returns running (m, l, acc) with m,l [B,H,L,1], acc [B,H,L,Dh] (NOT divided).
  """
  assert compare in ("causal", "strict"), (
      f"compare must be 'causal' or 'strict', got {compare!r}")
  B, L, H, Dh = q_BLHD.shape
  neg = jnp.finfo(jnp.float32).min
  qf = q_BLHD.astype(jnp.float32)
  kt_all, vt_all, kp_t, kvalid = _tile_keys(keys_BNHD, vals_BNHD, kp_N, block)
  m0 = jnp.full((B, H, L, 1), neg, jnp.float32)
  l0 = jnp.zeros((B, H, L, 1), jnp.float32)
  acc0 = jnp.zeros((B, H, L, Dh), jnp.float32)

  def body(carry, tile):
    m, l, acc = carry
    kt, vt, kpt, kv = tile                  # [B,block,H,Dh],..,[block],[block]
    s = jnp.einsum("bqhd,bkhd->bhqk", qf, kt.astype(jnp.float32))  # [B,H,L,blk]
    if compare == "causal":
      cmask = q_pos_L[:, None] >= kpt[None, :]
    else:
      cmask = q_pos_L[:, None] > kpt[None, :]
    cmask = cmask & kv[None, :]             # [L,block]
    mask = cmask[None, None]                # [1,1,L,block]
    if threshold is not None:
      mask = mask & (s >= threshold)        # [B,H,L,block]
    s_masked = jnp.where(mask, s, neg)
    m2 = jnp.maximum(m, jnp.max(s_masked, axis=-1, keepdims=True))
    corr = jnp.exp(m - m2)
    # Clamp masked logits to m2 so the exp argument is 0 for masked entries.
    # Using raw `s - m2` overflows to +inf when a tile is fully masked and
    # m2 == finfo.min (initial carry), and `where(False, +inf, 0)` then yields
    # 0*inf = NaN in the backward pass. The forward result is unchanged (masked
    # entries get weight 0 either way).
    s_safe = jnp.where(mask, s, m2)
    p = jnp.where(mask, jnp.exp(s_safe - m2), 0.0)       # [B,H,L,block]
    l = l * corr + jnp.sum(p, axis=-1, keepdims=True)
    acc = acc * corr + jnp.einsum("bhqk,bkhd->bhqd", p, vt.astype(jnp.float32))
    return (m2, l, acc), None

  (m, l, acc), _ = jax.lax.scan(
      jax.checkpoint(body), (m0, l0, acc0), (kt_all, vt_all, kp_t, kvalid))
  return m, l, acc


def _flash_topk_buffer(q_BLHD, keys_BNHD, kp_N, q_pos_L, compare, k, block):
  """Streaming top-k of the (causally-eligible) logits of one key set. Carries a
  per-query buffer of the best-k logits; merges each tile's top-k. Returns
  buf_logit [B,H,L,k] (NEG-padded when fewer than k valid). The k-th column
  (buf[..., -1:]) is the top-k threshold, matching naive `_topk_keep`'s `>=`."""
  assert compare in ("causal", "strict"), (
      f"compare must be 'causal' or 'strict', got {compare!r}")
  B, L, H, Dh = q_BLHD.shape
  neg = jnp.finfo(jnp.float32).min
  qf = q_BLHD.astype(jnp.float32)
  kt_all, _, kp_t, kvalid = _tile_keys(keys_BNHD, None, kp_N, block)
  buf0 = jnp.full((B, H, L, k), neg, jnp.float32)

  def body(buf, tile):
    kt, kpt, kv = tile
    s = jnp.einsum("bqhd,bkhd->bhqk", qf, kt.astype(jnp.float32))  # [B,H,L,blk]
    if compare == "causal":
      cmask = q_pos_L[:, None] >= kpt[None, :]
    else:
      cmask = q_pos_L[:, None] > kpt[None, :]
    cmask = cmask & kv[None, :]
    s = jnp.where(cmask[None, None], s, neg)
    kk = min(k, s.shape[-1])
    ts = jax.lax.top_k(s, kk)[0]                         # [B,H,L,kk]
    buf = jax.lax.top_k(jnp.concatenate([buf, ts], -1), k)[0]
    return buf, None

  buf, _ = jax.lax.scan(
      jax.checkpoint(body), buf0, (kt_all, kp_t, kvalid))
  return buf


def _merge(streams):
  """Online-softmax merge of several (m, l, acc) -> out [B,H,L,Dh] (divided).

  Precondition: for each query position at least one stream must have l > 0
  (>= 1 visible key); otherwise that row is 0/0 = NaN. In flash MoDA this holds
  because the row (causal self-attn) stream always sees the query's own position.
  """
  big_m = streams[0][0]
  for (m, _, _) in streams[1:]:
    big_m = jnp.maximum(big_m, m)
  l_tot = jnp.zeros_like(streams[0][1])
  acc_tot = jnp.zeros_like(streams[0][2])
  for (m, l, acc) in streams:
    c = jnp.exp(m - big_m)
    l_tot = l_tot + l * c
    acc_tot = acc_tot + acc * c
  return acc_tot / l_tot


def _flash_col(q_BLHD, depth_k, depth_v):
  """Same-position (col) depth contribution, computed directly (small: P*S keys
  per query). Returns (m, l, acc), all [B,H,L,*]."""
  B, L, H, Dh = q_BLHD.shape
  P, S = depth_k.shape[:2]
  qf = q_BLHD.astype(jnp.float32)
  col = jnp.einsum(
      "bqhd,psbqhd->bhqps", qf, depth_k.astype(jnp.float32)
  ).reshape(B, H, L, P * S)                            # [B,H,L,P*S]
  m = jnp.max(col, axis=-1, keepdims=True)
  p = jnp.exp(col - m)                                 # [B,H,L,P*S]
  l = jnp.sum(p, axis=-1, keepdims=True)
  acc = jnp.einsum(
      "bhqps,psbqhd->bhqd", p.reshape(B, H, L, P, S), depth_v.astype(jnp.float32))
  return m, l, acc


def _moda_attention_flash(q, k_cur, v_cur, depth_k, depth_v, cfg, gate):
  """Exact, memory-efficient MoDA attention via key-tile streaming. Numerically
  equal to the naive path for rest_select in {all, topk, global_topk} and
  two_stream (entmax is rejected upstream)."""
  B, L, H, Dh = q.shape
  P, S = depth_k.shape[:2]
  block = cfg.moda_flash_block
  q_pos = jnp.arange(L)
  key_pos = jnp.arange(L)
  sel = cfg.moda_rest_select

  def row(thr=None):
    return _flash_softmax_stream(q, k_cur, v_cur, key_pos, q_pos, "causal",
                                 block, thr)

  if P == 0:
    m, l, acc = row()
    return (acc / l).transpose(0, 2, 1, 3)

  dk, dv = _flatten_depth_kv(depth_k, depth_v)          # [B,M,H,Dh]
  depth_time = jnp.tile(jnp.arange(L)[None, :], (P * S, 1)).reshape(-1)  # [M]

  def depth(compare, thr=None):
    return _flash_softmax_stream(q, dk, dv, depth_time, q_pos, compare, block,
                                 thr)

  if cfg.moda_two_stream:
    out_A = _merge([row(), _flash_col(q, depth_k, depth_v)])   # [B,H,L,Dh]
    if sel == "topk":
      thr = _flash_topk_buffer(
          q, dk, depth_time, q_pos, "strict", cfg.moda_rest_topk, block)[
              ..., -1:]
      m_b, l_b, acc_b = depth("strict", thr)
    else:  # all
      m_b, l_b, acc_b = depth("strict")
    has_rest = l_b > 0.0                                       # [B,H,L,1]
    out_B = acc_b / jnp.where(has_rest, l_b, 1.0)
    g = jax.nn.sigmoid(gate.astype(jnp.float32)).reshape(1, H, 1, 1)
    g = jnp.where(has_rest, g, jnp.ones((), jnp.float32))      # force A if no rest
    out = g * out_A + (1.0 - g) * out_B
    return out.transpose(0, 2, 1, 3)

  k = cfg.moda_rest_topk
  if sel == "all":
    out = _merge([row(), depth("causal")])
  elif sel == "topk":
    thr = _flash_topk_buffer(q, dk, depth_time, q_pos, "strict", k, block)[
        ..., -1:]
    out = _merge([row(), _flash_col(q, depth_k, depth_v), depth("strict", thr)])
  elif sel == "global_topk":
    buf_row = _flash_topk_buffer(q, k_cur, key_pos, q_pos, "causal", k, block)
    buf_dep = _flash_topk_buffer(q, dk, depth_time, q_pos, "causal", k, block)
    thr = jax.lax.top_k(jnp.concatenate([buf_row, buf_dep], -1), k)[0][..., -1:]
    out = _merge([row(thr), depth("causal", thr)])
  else:  # entmax reaches here only via direct call; routing rejects it.
    raise NotImplementedError(f"flash does not support rest_select={sel!r}")
  return out.transpose(0, 2, 1, 3)


class MoDAAttn(nn.Module):
  """MoDA attention with shared projections and selectable attention dataflow."""

  cfg: DepthMemConfig
  layer_index: int

  @nn.compact
  def __call__(self, x_BxLxD, mem_k, mem_v):
    cfg = self.cfg
    _assert_heads_divide_dim(cfg)
    Dh = cfg.D // cfg.H

    multilinear = partial(
        nn.DenseGeneral,
        axis=-1,
        features=(cfg.H, Dh),
        kernel_init=fsdp.init("attn_in_proj", cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )

    q_BxLxHxDh = multilinear(name="query")(x_BxLxD) / (Dh ** 0.5)
    k_cur = multilinear(name="key")(x_BxLxD)
    v_cur = multilinear(name="value")(x_BxLxD)

    P = self.layer_index
    depth_k = mem_k[:P]
    depth_v = mem_v[:P]

    uses_select = cfg.moda_two_stream or cfg.moda_rest_select != "all"
    if cfg.moda_impl == "flash":
      assert cfg.moda_depth_attn_mode == "causal", (
          "moda_impl='flash' requires moda_depth_attn_mode='causal'")
      if cfg.moda_rest_select == "entmax":
        raise NotImplementedError(
            "entmax is not flash-able; use moda_impl='naive'")
      if cfg.moda_rest_select in ("topk", "global_topk"):
        assert cfg.moda_rest_topk > 0, (
            "moda_rest_topk must be > 0 when "
            f"moda_rest_select={cfg.moda_rest_select!r}")
      if cfg.moda_rest_select == "global_topk":
        assert not cfg.moda_two_stream, (
            "moda_rest_select='global_topk' is single-stream only")
      gate = None
      if cfg.moda_two_stream:
        gate = self.param(
            "two_stream_gate",
            lambda key: jnp.full(
                (cfg.H,), cfg.moda_two_stream_gate_init, cfg.dtype),
        )
      out_BxLxHxDh = _moda_attention_flash(
          q_BxLxHxDh, k_cur, v_cur, depth_k, depth_v, cfg, gate)
    elif uses_select:
      assert cfg.moda_depth_attn_mode == "causal", (
          "moda_rest_select / moda_two_stream require "
          "moda_depth_attn_mode='causal'")
      if cfg.moda_impl != "naive":
        raise NotImplementedError(
            "moda_rest_select / moda_two_stream support moda_impl='naive' only")
      if cfg.moda_rest_select in ("topk", "global_topk"):
        assert cfg.moda_rest_topk > 0, (
            "moda_rest_topk must be > 0 when "
            f"moda_rest_select={cfg.moda_rest_select!r}")
      if cfg.moda_rest_select == "global_topk":
        assert not cfg.moda_two_stream, (
            "moda_rest_select='global_topk' is single-stream only "
            "(it selects over the whole grid); use moda_two_stream=False")
      gate = None
      if cfg.moda_two_stream:
        gate = self.param(
            "two_stream_gate",
            lambda key: jnp.full(
                (cfg.H,), cfg.moda_two_stream_gate_init, cfg.dtype),
        )
      out_BxLxHxDh = _moda_attention_naive_select(
          q_BxLxHxDh, k_cur, v_cur, depth_k, depth_v, cfg, gate)
    else:
      if cfg.moda_impl == "naive":
        chunk_size = x_BxLxD.shape[1]
      elif cfg.moda_impl == "chunked":
        chunk_size = cfg.moda_chunk_size
      else:
        raise NotImplementedError(f"unknown moda_impl {cfg.moda_impl!r}")
      out_BxLxHxDh = _moda_attention_chunked(
          q_BxLxHxDh,
          k_cur,
          v_cur,
          depth_k,
          depth_v,
          chunk_size,
          cfg.moda_depth_attn_mode,
          cfg.dtype,
      )

    out_BxLxD = nn.DenseGeneral(
        features=cfg.D,
        name="attn_out_proj",
        axis=(-2, -1),
        kernel_init=fsdp.init("attn_out_proj", cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )(out_BxLxHxDh)
    return out_BxLxD, k_cur, v_cur


_DCA_MLP_READ_MODES = ("none", "grn_pool", "grn_pool_res_in", "grn_pool_res_out")


class DepthGRNMlp(nn.Module):
  """MLP sub-block whose input optionally reads a depth-history GRN pool.

  Reusable across DCA variants: the caller passes the residual tensor (the
  running hidden state `x` for post_proj DCA) and a `[N, B, L, D]` buffer of
  past stored slots. `cfg.dca_mlp_read` selects how the MLP input is built
  (shown pre-RMSNorm); `G(.)` is a 1-output GRN pool over
  `[past stored slots | current slot]`, with `a` = `attn_out`:

    "none"             pre = residual + a              (no pool)
    "grn_pool"         pre = G([hist, a])              (cur = a)
    "grn_pool_res_in"  pre = G([hist, residual + a])   (cur folds residual in)
    "grn_pool_res_out" pre = G([hist, a]) + residual   (residual added outside)

  The current slot is always pooled, so the GRN can gate it (unlike the old
  `grn_mlp_mem`, which added the current attention output outside the pool).

  Returns `(ff_delta, pre_ff, new_slot)`. `new_slot` is what to write to the
  history buffer at this layer's slot (None for "none"); it is `residual + a`
  for res_in and `a` for grn_pool / res_out.
  """

  cfg: DepthMemConfig
  layer_index: int

  @nn.compact
  def __call__(self, mlp_history, attn_out_BxLxD, residual_BxLxD):
    cfg = self.cfg
    mode = cfg.dca_mlp_read
    assert mode in _DCA_MLP_READ_MODES, f"unknown dca_mlp_read {mode!r}"

    norm = nn.RMSNorm(dtype=cfg.dtype, name="pre_ff_norm")
    mlp = base_model.Mlp(cfg, name="mlp")

    if mode == "none":
      pre_ff = norm(residual_BxLxD + attn_out_BxLxD)
      return mlp(pre_ff), pre_ff, None

    cur_BxLxD = (
        residual_BxLxD + attn_out_BxLxD
        if mode == "grn_pool_res_in"
        else attn_out_BxLxD
    )
    # [past slots | current] -> 1-output GRN pool. layer_index past slots plus
    # the current slot; current tagged 2*N so analysis can split it from past.
    full = jnp.concatenate(
        [mlp_history[: self.layer_index], cur_BxLxD[None]], axis=0
    )
    (agg,) = GRNReader(
        cfg,
        num_layers_visible=self.layer_index + 1,
        num_outputs=1,
        source_layer_indices=tuple(range(self.layer_index)) + (2 * cfg.N,),
        name="grn_mlp",
    )(full)
    pre = agg + residual_BxLxD if mode == "grn_pool_res_out" else agg
    pre_ff = norm(pre)
    return mlp(pre_ff), pre_ff, cur_BxLxD


class DepthMemTBlock(nn.Module):
  """Transformer block with variant-specific depth-memory read/write."""

  cfg: DepthMemConfig
  layer_index: int

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array, memory):
    cfg = self.cfg
    if cfg.variant == "dca":
      if _dca_uses_post_proj(cfg):
        return self._call_dca_post_proj(x_BxLxD, memory)
      return self._call_dca(x_BxLxD, memory)
    if cfg.variant == "mhc":
      return self._call_mhc(x_BxLxD, memory)
    if cfg.variant == "moda":
      return self._call_moda(x_BxLxD, memory)
    raise NotImplementedError(f"variant {cfg.variant!r} not implemented")

  def _call_dca(self, x_BxLxD, memory):
    cfg = self.cfg
    del x_BxLxD  # DCA reads from `memory` directly; current x lives at memory[layer_index].

    valid_K = self.layer_index + 1
    mem_visible, source_indices = _dca_select(
        memory[:valid_K], cfg.dca_select_mode, cfg.past_layers_k
    )
    K_after_select = len(source_indices)

    q_in, k_in, v_in = GRNReader(
        cfg,
        num_layers_visible=K_after_select,
        num_outputs=3,
        source_layer_indices=source_indices,
    )(mem_visible)
    # NOTE: `residual = q_in` is only used to stabilize pre_ff_norm's input;
    # it deliberately does NOT appear in `block_out` below. This matches
    # lucidrains/deep-cross-attention's DCABlock.forward exactly:
    #   residual = q_input
    #   attn_out = self.attn(q_input, k_input, v_input)
    #   ff_input = self.pre_ff_norm(attn_out + residual)
    #   ff_out   = self.ff(ff_input)
    #   return ff_out + attn_out
    # Unlike a standard pre-norm transformer, DCA does not add the per-block
    # residual into the block's output. The "residual stream" in DCA is the
    # depth-axis memory: q_in is already present in some past slot of
    # `memory` (most recently slot `layer_index`), and the next block's GRN
    # can re-mix it via its layer gating. Adding it into `block_out` here
    # would double-count that contribution.
    residual = q_in
    attn_out = base_model.CausalAttn(cfg, name="attn")(
        nn.RMSNorm(dtype=cfg.dtype, name="q_norm")(q_in),
        nn.RMSNorm(dtype=cfg.dtype, name="k_norm")(k_in),
        nn.RMSNorm(dtype=cfg.dtype, name="v_norm")(v_in),
    )
    pre_ff = nn.RMSNorm(dtype=cfg.dtype, name="pre_ff_norm")(
        attn_out + residual
    )
    ff_out = base_model.Mlp(cfg, name="mlp")(pre_ff)
    block_out = ff_out + attn_out  # intentionally no `+ residual`; see above
    new_memory = memory.at[self.layer_index + 1].set(block_out)
    # The first return slot is the per-block "hidden state" used by MoDA; for
    # DCA the next block reads `memory` directly, so we return a 0-d sentinel
    # that JIT/XLA will DCE. Using a real jax array (not None) keeps the tuple
    # well-typed for nn.remat compatibility.
    return _DEAD_VALUE(cfg.dtype), new_memory

  def _call_dca_post_proj(self, x_BxLxD, memory):
    """DCA where the depth aggregation happens AFTER QKV projection.

    Each axis (Q, K, V) is independently toggleable via `cfg.dca_mix_{q,k,v}`.
    For each enabled axis, the current block's projection is concatenated
    with the past blocks' projections of the same axis and aggregated by a
    per-head `PerHeadGRNReader`. Axes that are off use the current block's
    projection directly (no cross-layer mix on that axis).

    Memory is a dict — keys present only for enabled axes/modes:

      "k"     : [N, B, L, H, Dh]  — past attn K (iff dca_mix_k)
      "v"     : [N, B, L, H, Dh]  — past attn V (iff dca_mix_v)
      "q"     : [N, B, L, H, Dh]  — past attn Q (iff dca_mix_q)
      "k_mlp" : [N, B, L, H, Dh]  — past MLP-written K (iff write==moda_kv & mix_k)
      "v_mlp" : [N, B, L, H, Dh]  — past MLP-written V (iff write==moda_kv & mix_v)
      "mlp"   : [N, B, L, D]      — past MLP slots (iff dca_mlp_read != "none")

    The MLP is delegated to `DepthGRNMlp`. Its input (read axis) is controlled
    by `cfg.dca_mlp_read`, and the optional MoDA-style K/V write by the
    orthogonal `cfg.dca_mlp_write`; see the config docstring for semantics.
    """
    cfg = self.cfg
    _assert_heads_divide_dim(cfg)
    Dh = cfg.D // cfg.H
    mix_q, mix_k, mix_v = cfg.dca_mix_q, cfg.dca_mix_k, cfg.dca_mix_v

    # ---------- Q/K/V projection of the current hidden state ----------
    multilinear = partial(
        nn.DenseGeneral,
        axis=-1,
        features=(cfg.H, Dh),
        kernel_init=fsdp.init("attn_in_proj", cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )
    x_norm = nn.RMSNorm(dtype=cfg.dtype, name="pre_attn_norm")(x_BxLxD)
    q_cur = multilinear(name="query")(x_norm) / (Dh ** 0.5)
    k_cur = multilinear(name="key")(x_norm)
    v_cur = multilinear(name="value")(x_norm)

    # ---------- Per-axis GRN aggregation (enabled axes only) ----------
    def _aggregate(name, current, past_attn, past_mlp=None):
      """Build [past_attn | (optional past_mlp) | current] and run per-head GRN."""
      stack = [past_attn[: self.layer_index]]
      indices = list(range(self.layer_index))
      if past_mlp is not None:
        stack.append(past_mlp[: self.layer_index])
        # MLP-written slots tagged with offset N so analysis can split them.
        indices.extend(cfg.N + i for i in range(self.layer_index))
      stack.append(current[None])
      indices.append(2 * cfg.N)  # sentinel: current layer's own projection.
      full = jnp.concatenate(stack, axis=0)
      (mixed,) = PerHeadGRNReader(
          cfg,
          num_layers_visible=full.shape[0],
          num_outputs=1,
          source_layer_indices=tuple(indices),
          name=f"grn_{name}",
      )(full)
      return mixed

    # Pass mem_k_mlp / mem_v_mlp only when the write axis is moda_kv AND the
    # corresponding axis is mixed (otherwise that depth slot isn't
    # populated, see write-side at bottom of method).
    moda_kv = cfg.dca_mlp_write == "moda_kv"
    q_mixed = _aggregate("q", q_cur, memory["q"]) if mix_q else q_cur
    k_mixed = _aggregate(
        "k", k_cur, memory["k"],
        memory["k_mlp"] if (moda_kv and mix_k) else None,
    ) if mix_k else k_cur
    v_mixed = _aggregate(
        "v", v_cur, memory["v"],
        memory["v_mlp"] if (moda_kv and mix_v) else None,
    ) if mix_v else v_cur

    # ---------- Standard causal attention ----------
    L = x_BxLxD.shape[1]
    logits = jnp.einsum(
        "bqhd,bkhd->bhqk", q_mixed, k_mixed
    ).astype(jnp.float32)
    mask = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(logits, axis=-1).astype(cfg.dtype)
    attn_pre = jnp.einsum("bhqk,bkhd->bqhd", weights, v_mixed)
    attn_delta = nn.DenseGeneral(
        features=cfg.D,
        axis=(-2, -1),
        name="attn_out_proj",
        kernel_init=fsdp.init("attn_out_proj", cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )(attn_pre)
    x_post_attn = x_BxLxD + attn_delta

    # ---------- MLP path (read axis) ----------
    # Residual is the block input `x`; attention output is `attn_delta`. For
    # read="none" this is exactly the standard pre-FF norm(x + attn_delta).
    ff_delta, pre_ff, mlp_slot = DepthGRNMlp(
        cfg, layer_index=self.layer_index, name="mlp_block"
    )(memory.get("mlp"), attn_delta, x_BxLxD)
    block_out = x_post_attn + ff_delta

    # ---------- Write to memory ----------
    new_memory = dict(memory)  # shallow copy; we'll overwrite enabled keys.

    if mix_q:
      new_memory["q"] = memory["q"].at[self.layer_index].set(q_cur)
    if mix_k:
      new_memory["k"] = memory["k"].at[self.layer_index].set(k_cur)
    if mix_v:
      new_memory["v"] = memory["v"].at[self.layer_index].set(v_cur)

    # Write axis: MoDA-style K/V slot from the MLP input, for future blocks.
    if moda_kv and (mix_k or mix_v) and self.layer_index + 1 < cfg.N:
      # Last-block MLP write skipped (no future read).
      mlp_k_cur, mlp_v_cur = MoDAKVProj(cfg, name="mlp_depth_kv")(pre_ff)
      if mix_k:
        new_memory["k_mlp"] = memory["k_mlp"].at[self.layer_index].set(mlp_k_cur)
      if mix_v:
        new_memory["v_mlp"] = memory["v_mlp"].at[self.layer_index].set(mlp_v_cur)

    # Read axis: store this block's MLP slot for later blocks' GRN pools.
    if cfg.dca_mlp_read != "none":
      new_memory["mlp"] = memory["mlp"].at[self.layer_index].set(mlp_slot)

    return block_out, new_memory

  def _call_mhc(self, x_BxLxD, memory):
    cfg = self.cfg
    del x_BxLxD  # mHC: streams ARE the state; no separate hidden-state input.
    streams = memory
    is_identity = cfg.mhc_h_res_mode == "identity"

    h_pre, h_post, h_res = MHCMappings(cfg, name="map_attn")(streams)
    agg = _mhc_aggregate(streams, h_pre)
    normed = nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="ln_attn")(agg)
    f_out = base_model.CausalAttn(cfg, name="attn")(normed)
    streams = _mhc_step(
        streams, h_post, h_res, f_out, h_res_is_identity=is_identity,
    )

    h_pre, h_post, h_res = MHCMappings(cfg, name="map_mlp")(streams)
    agg = _mhc_aggregate(streams, h_pre)
    normed = nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="ln_mlp")(agg)
    f_out = base_model.Mlp(cfg, name="mlp")(normed)
    streams = _mhc_step(
        streams, h_post, h_res, f_out, h_res_is_identity=is_identity,
    )

    return _DEAD_VALUE(cfg.dtype), streams

  def _call_moda(self, x_BxLxD, memory):
    cfg = self.cfg
    mem_k, mem_v = memory

    x_norm = nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="ln_attn")(
        x_BxLxD
    )
    attn_out, attn_k, attn_v = MoDAAttn(
        cfg, layer_index=self.layer_index, name="moda_attn"
    )(x_norm, mem_k, mem_v)
    x = x_BxLxD + attn_out

    mlp_input = nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="ln_mlp")(x)
    mlp_out = base_model.Mlp(cfg, name="mlp")(mlp_input)
    out = x + mlp_out

    if cfg.moda_include_mlp_kv and self.layer_index + 1 < cfg.N:
      mlp_k, mlp_v = MoDAKVProj(cfg, name="mlp_depth_kv")(mlp_input)
      mem_k = mem_k.at[self.layer_index].set(
          jnp.stack([attn_k, mlp_k], axis=0)
      )
      mem_v = mem_v.at[self.layer_index].set(
          jnp.stack([attn_v, mlp_v], axis=0)
      )
    else:
      mem_k = mem_k.at[self.layer_index, 0].set(attn_k)
      mem_v = mem_v.at[self.layer_index, 0].set(attn_v)
    return out, (mem_k, mem_v)


_DEPTH_MEM_VARIANTS = ("dca", "mhc", "moda")


def _embed_with_pos(
    embed: nn.Embed, pos_embed: nn.Embed, y_BxL: jax.Array
) -> jax.Array:
  """Token embedding + learned positional embedding (mirrors `TransformerDo`)."""
  y_BxLxD = embed(y_BxL)
  y_BxLxD += pos_embed(jnp.arange(0, y_BxL.shape[1])[None, ...])
  return y_BxLxD


def _project_to_logits(
    out_ln: nn.Module, embed: nn.Embed, hidden_BxLxD: jax.Array
) -> jax.Array:
  """Final norm + tied-embedding logits (fp32) — shared by every variant."""
  return embed.attend(out_ln(hidden_BxLxD).astype(jnp.float32))


class DepthMemTransformerDo(nn.Module):
  """Decoder-only Transformer with depth-memory variants.

  Supports `variant in {"dca", "mhc", "moda"}`. The baseline transformer
  lives in `nanodo.model.TransformerDo`; `model_factory.get_model_and_loss`
  routes `variant == "baseline"` there directly so this class never has to
  carry that path.
  """

  docfg: DepthMemConfig

  def setup(self):
    cfg = self.docfg
    assert cfg.variant in _DEPTH_MEM_VARIANTS, (
        f"DepthMemTransformerDo only supports {_DEPTH_MEM_VARIANTS}; "
        f"got {cfg.variant!r}. The model_factory routes 'baseline' to "
        f"nanodo.model.TransformerDo."
    )

    self.embed = nn.Embed(
        num_embeddings=cfg.V,
        features=cfg.D,
        embedding_init=fsdp.init("embedding", cfg),
    )
    self.pos_embed = nn.Embed(
        num_embeddings=cfg.L,
        features=cfg.D,
        embedding_init=fsdp.init("embedding", cfg),
    )
    if cfg.variant == "dca":
      self.out_ln = nn.RMSNorm(dtype=cfg.dtype)
    else:
      self.out_ln = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)

    block_cls = nn.remat(DepthMemTBlock) if cfg.remat else DepthMemTBlock
    self.blocks = [block_cls(cfg, layer_index=i) for i in range(cfg.N)]

    if cfg.variant == "dca" and not _dca_uses_post_proj(cfg):
      # final_grn only exists for hidden-DCA (pool over the full memory at
      # the end). The post_proj path uses the last block's hidden state
      # directly with no final pool.
      final_indices = _dca_source_indices(
          cfg.N + 1, cfg.dca_select_mode, cfg.past_layers_k
      )
      self.final_grn = GRNReader(
          cfg,
          num_layers_visible=len(final_indices),
          num_outputs=1,
          source_layer_indices=final_indices,
      )

  def __call__(self, y_BxL: jax.Array):
    cfg = self.docfg
    B, L = y_BxL.shape

    y_BxLxD = _embed_with_pos(self.embed, self.pos_embed, y_BxL)

    if cfg.variant == "dca" and not _dca_uses_post_proj(cfg):
      memory = jnp.zeros((cfg.N + 1, B, L, cfg.D), dtype=_cfg_dtype(cfg))
      memory = memory.at[0].set(y_BxLxD)
      for block in self.blocks:
        _, memory = block(None, memory)
      mem_visible, _ = _dca_select(
          memory, cfg.dca_select_mode, cfg.past_layers_k
      )
      (pooled,) = self.final_grn(mem_visible)
      return _project_to_logits(self.out_ln, self.embed, pooled)

    if cfg.variant == "dca" and _dca_uses_post_proj(cfg):
      _assert_heads_divide_dim(cfg)
      Dh = cfg.D // cfg.H
      per_head_shape = (cfg.N, B, L, cfg.H, Dh)
      dtype = _cfg_dtype(cfg)
      # Allocate one [N, B, L, H, Dh] buffer per enabled axis. Disabled axes
      # use the current block's projection directly — no memory needed.
      memory = {}
      if cfg.dca_mix_q:
        memory["q"] = jnp.zeros(per_head_shape, dtype=dtype)
      if cfg.dca_mix_k:
        memory["k"] = jnp.zeros(per_head_shape, dtype=dtype)
      if cfg.dca_mix_v:
        memory["v"] = jnp.zeros(per_head_shape, dtype=dtype)
      # Write axis (moda_kv): per-head K/V slots written by the MLP.
      if cfg.dca_mlp_write == "moda_kv":
        if cfg.dca_mix_k:
          memory["k_mlp"] = jnp.zeros(per_head_shape, dtype=dtype)
        if cfg.dca_mix_v:
          memory["v_mlp"] = jnp.zeros(per_head_shape, dtype=dtype)
      # Read axis (grn_pool*): D-space buffer of past MLP slots (no embedding
      # slot — slot i is block i's stored slot; res_in folds the embedding in
      # via block 0's residual).
      if cfg.dca_mlp_read != "none":
        memory["mlp"] = jnp.zeros((cfg.N, B, L, cfg.D), dtype=dtype)
      last = y_BxLxD
      for block in self.blocks:
        last, memory = block(last, memory)
      return _project_to_logits(self.out_ln, self.embed, last)

    if cfg.variant == "mhc":
      n = cfg.num_streams
      streams = jnp.broadcast_to(
          y_BxLxD[:, :, None, :], (B, L, n, cfg.D)
      )
      streams = jnp.asarray(streams)
      for block in self.blocks:
        # _call_mhc ignores the first arg (`del x_BxLxD`); pass None so we
        # don't compute a wasted `streams.mean(axis=2)` per block. The
        # block returns a 0-d dead sentinel which we discard.
        _, streams = block(None, streams)
      return _project_to_logits(self.out_ln, self.embed, streams.mean(axis=2))

    if cfg.variant == "moda":
      _assert_heads_divide_dim(cfg)
      slots = 2 if cfg.moda_include_mlp_kv else 1
      Dh = cfg.D // cfg.H
      mem_k = jnp.zeros((cfg.N, slots, B, L, cfg.H, Dh), dtype=_cfg_dtype(cfg))
      mem_v = jnp.zeros_like(mem_k)
      memory = (mem_k, mem_v)
      last = y_BxLxD
      for block in self.blocks:
        last, memory = block(last, memory)
      return _project_to_logits(self.out_ln, self.embed, last)

    raise AssertionError(  # unreachable: setup() already validated
        f"variant {cfg.variant!r} fell through the dispatch"
    )
