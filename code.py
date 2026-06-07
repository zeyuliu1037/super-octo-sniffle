# Copyright 2026.
"""Scheduled Depth Routing / Routed Depth Memory variants.

This module implements the first RDM path from
`nanodo/routed_depth_memory_complete_plan.md`: scheduled writer-side depth
materialization with a DCA-style soft reader. The base block follows
`nanodo.model` with the RoPE / QK-norm attention conventions from
`nanodo.variants.mod`.
"""

# pylint: disable=invalid-name,g-importing-member

import dataclasses
from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp

from nanodo import fsdp
from nanodo import model
from nanodo.variants import dca_ref


_STREAMS_QKVR = ("q", "k", "v", "r")


@dataclasses.dataclass
class RdmConfig(model.DoConfig):
  """Config for Scheduled Depth Routing / Routed Depth Memory."""

  variant: str = "rdm"

  # Base attention.
  use_rope: bool = True
  rope_theta: float = 10000.0
  qk_norm: bool = False
  router_type: str = "linear"  # {"linear", "mlp"}

  # Depth backend.
  rdm_backend: str = "dca"  # {"dca", "hc", "attnres"}
  rdm_mix_mode: str = "hidden"  # {"hidden", "qkv", "qkvr"}
  rdm_block_style: str = "transformer"  # {"transformer", "dca"}
  rdm_source_state: str = "hidden"  # {"hidden", "delta", "ln_hidden", "ln_delta"}
  rdm_include_immediate_prev: bool = True
  rdm_include_embedding: bool = True
  # Number of first K/V attention heads that use routed depth memory in qkv/qkvr.
  # -1 => all heads; 0 => K/V fully use the current layer input.
  rdm_headwise_kv_heads: int = -1
  # "safe" chooses an edge init compatible with a near-vanilla residual path.
  rdm_init_mode: str = "safe"  # {"safe", "immediate", "dense", "closed"}
  rdm_reader_type: str = "grn_v3"

  # Writer side.
  rdm_writer_enabled: bool = True
  rdm_writer_version: str = "targeted"  # {"simple", "targeted"}
  rdm_writer_streams: str = "shared"  # {"shared", "qkv", "qkvr", "hc_stream"}
  rdm_writer_edge_apply: str = "logit_prior"  # {"logit_prior", "mask_only", "value_scale", "inbox_scale"}
  rdm_temperature: float = 2.0
  rdm_prior_scale: float = 1.0
  rdm_prior_epsilon: float = 1e-6
  rdm_writer_rank: int = 16
  rdm_init_open_logit: float = 20.0
  rdm_init_closed_logit: float = -20.0

  # Hard write / budget controls.
  rdm_write_hard: bool = False
  rdm_hardening: str = "none"  # {"none", "st_threshold", "st_source_topk", "st_target_topk"}
  rdm_threshold: float = 0.5
  rdm_k_out: int = 4
  rdm_k_in: int = 8

  # Aux losses.
  rdm_align_weight: float = 0.01
  rdm_z_weight: float = 0.0

  # HC / mHC backend.
  num_streams: int = 4
  sinkhorn_iters: int = 20
  mhc_init_gating_factor: float = 0.01
  mhc_h_res_mode: str = "sinkhorn"  # {"sinkhorn", "identity"}

  # AttnRes backend.
  attnres_variant: str = "full"  # {"full", "block"}
  attnres_block_size: int = 4

  # DCA selector compatibility. `experimental_model` must be truthy for the
  # original `dca_ref.InputSelector` path; this module mirrors that selector.
  experimental_model: str = "rdm"
  dca_grn_topk: float = 0.0


def _rope_sincos(positions_BxL, Dh, theta, dtype):
  """Return (sin, cos) of shape [B, L, Dh//2] for rotary embedding."""
  half = Dh // 2
  inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
  inv_freq = inv_freq.reshape(1, 1, half)
  angles = positions_BxL[..., None].astype(jnp.float32) * inv_freq
  return jnp.sin(angles).astype(dtype), jnp.cos(angles).astype(dtype)


def _apply_rope(x_BxLxHxDh, sin_BxLxhalf, cos_BxLxhalf):
  """Apply rotate-half RoPE to a per-head tensor."""
  half = x_BxLxHxDh.shape[-1] // 2
  x1 = x_BxLxHxDh[..., :half]
  x2 = x_BxLxHxDh[..., half:]
  sin = sin_BxLxhalf[:, :, None, :]
  cos = cos_BxLxhalf[:, :, None, :]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _stream_count(cfg: RdmConfig) -> int:
  if cfg.rdm_writer_streams == "shared":
    return 1
  if cfg.rdm_writer_streams == "qkv":
    return 3
  if cfg.rdm_writer_streams == "qkvr":
    return 4
  if cfg.rdm_writer_streams == "hc_stream":
    return cfg.num_streams
  raise ValueError(f"unknown rdm_writer_streams {cfg.rdm_writer_streams!r}")


def _validate_cfg(cfg: RdmConfig):
  if cfg.rdm_backend not in ("dca", "hc", "attnres"):
    raise ValueError(f"unknown rdm_backend {cfg.rdm_backend!r}")
  if cfg.rdm_backend == "dca" and cfg.rdm_reader_type != "grn_v3":
    raise NotImplementedError(
        f"rdm_reader_type={cfg.rdm_reader_type!r}; only 'grn_v3' is implemented")
  if cfg.rdm_mix_mode not in ("hidden", "qkv", "qkvr"):
    raise ValueError(f"unknown rdm_mix_mode {cfg.rdm_mix_mode!r}")
  if cfg.rdm_block_style not in ("transformer", "dca"):
    raise ValueError(f"unknown rdm_block_style {cfg.rdm_block_style!r}")
  if cfg.rdm_block_style == "dca":
    if cfg.rdm_backend != "dca" or cfg.rdm_mix_mode != "qkv":
      raise NotImplementedError(
          "rdm_block_style='dca' requires rdm_backend='dca' and "
          "rdm_mix_mode='qkv'")
    if cfg.rdm_source_state != "hidden":
      raise NotImplementedError(
          "rdm_block_style='dca' requires rdm_source_state='hidden'")
    if not cfg.rdm_include_embedding:
      raise NotImplementedError(
          "rdm_block_style='dca' requires rdm_include_embedding=True")
    if cfg.rdm_headwise_kv_heads != -1:
      raise NotImplementedError(
          "rdm_block_style='dca' currently requires all KV heads to use memory")
  if cfg.rdm_source_state not in ("hidden", "delta", "ln_hidden", "ln_delta"):
    raise ValueError(f"unknown rdm_source_state {cfg.rdm_source_state!r}")
  if cfg.rdm_writer_version not in ("simple", "targeted"):
    raise ValueError(f"unknown rdm_writer_version {cfg.rdm_writer_version!r}")
  if cfg.rdm_writer_streams not in ("shared", "qkv", "qkvr", "hc_stream"):
    if cfg.rdm_writer_streams == "headwise":
      raise NotImplementedError(
          "rdm_writer_streams='headwise' is not implemented; use shared, "
          "qkv, qkvr, or hc_stream")
    raise ValueError(f"unknown rdm_writer_streams {cfg.rdm_writer_streams!r}")
  if cfg.rdm_writer_edge_apply not in (
      "logit_prior", "mask_only", "value_scale", "inbox_scale"):
    raise ValueError(
        f"unknown rdm_writer_edge_apply {cfg.rdm_writer_edge_apply!r}")
  if cfg.rdm_init_mode not in ("safe", "immediate", "dense", "closed"):
    raise ValueError(f"unknown rdm_init_mode {cfg.rdm_init_mode!r}")
  if cfg.rdm_hardening not in (
      "none", "st_threshold", "st_source_topk", "st_target_topk"):
    raise ValueError(f"unknown rdm_hardening {cfg.rdm_hardening!r}")
  if cfg.rdm_init_mode == "safe" and cfg.rdm_writer_enabled:
    if cfg.rdm_writer_version != "targeted":
      raise NotImplementedError(
          "rdm_init_mode='safe' requires rdm_writer_version='targeted'")
    if cfg.rdm_backend in ("dca", "attnres") and (
        not cfg.rdm_include_immediate_prev):
      raise NotImplementedError(
          "rdm_init_mode='safe' requires rdm_include_immediate_prev=True")
    if cfg.rdm_backend == "attnres" and cfg.rdm_source_state != "hidden":
      raise NotImplementedError(
          "rdm_init_mode='safe' for attnres requires rdm_source_state='hidden'")
    if cfg.rdm_backend == "dca" and cfg.rdm_source_state not in (
        "hidden", "delta"):
      raise NotImplementedError(
          "rdm_init_mode='safe' for dca supports hidden or delta source states")
    if (cfg.rdm_backend == "dca" and cfg.rdm_source_state == "delta"
        and not cfg.rdm_include_embedding):
      raise NotImplementedError(
          "rdm_init_mode='safe' with DCA delta messages requires embedding")
  if cfg.rdm_headwise_kv_heads < -1:
    raise ValueError("rdm_headwise_kv_heads must be >= -1")
  if cfg.rdm_headwise_kv_heads != -1 and cfg.rdm_mix_mode == "hidden":
    raise NotImplementedError(
        "rdm_headwise_kv_heads only supports rdm_mix_mode='qkv' or 'qkvr'")
  if cfg.rdm_backend == "hc" and cfg.rdm_mix_mode != "hidden":
    raise NotImplementedError("rdm_backend='hc' currently supports hidden mode")
  if cfg.rdm_backend == "hc" and cfg.mhc_h_res_mode not in ("sinkhorn", "identity"):
    raise ValueError(f"unknown mhc_h_res_mode {cfg.mhc_h_res_mode!r}")
  if cfg.rdm_backend == "hc" and cfg.rdm_writer_streams not in (
      "shared", "hc_stream"):
    raise ValueError(
        "rdm_backend='hc' requires rdm_writer_streams='shared' or 'hc_stream'")
  if (cfg.rdm_backend == "hc" and cfg.rdm_writer_enabled
      and cfg.rdm_writer_edge_apply not in (
          "value_scale", "inbox_scale", "mask_only")):
    raise ValueError(
        "rdm_backend='hc' has no reader logits; use "
        "rdm_writer_edge_apply='value_scale', 'inbox_scale', or 'mask_only'")
  if cfg.rdm_backend == "attnres" and cfg.attnres_variant not in (
      "full", "block"):
    raise ValueError(f"unknown attnres_variant {cfg.attnres_variant!r}")
  if cfg.rdm_backend == "attnres" and cfg.attnres_block_size <= 0:
    raise ValueError("attnres_block_size must be > 0")
  if (cfg.rdm_backend != "hc" and cfg.rdm_mix_mode == "hidden"
      and cfg.rdm_writer_streams != "shared"):
    raise ValueError("rdm_mix_mode='hidden' requires rdm_writer_streams='shared'")
  if cfg.rdm_backend != "hc" and cfg.rdm_writer_streams == "hc_stream":
    raise ValueError("rdm_writer_streams='hc_stream' requires rdm_backend='hc'")
  if cfg.rdm_mix_mode == "qkv" and cfg.rdm_writer_streams == "qkvr":
    raise ValueError("rdm_mix_mode='qkv' does not consume an R writer stream")
  if cfg.rdm_mix_mode == "qkvr" and cfg.rdm_writer_streams == "qkv":
    raise ValueError("rdm_mix_mode='qkvr' requires shared or qkvr writer streams")


def _one_hot_topk_mask(scores, k: int, axis: int, valid_mask=None):
  """Boolean exact top-k mask along `axis`, optionally intersected with valid."""
  K = scores.shape[axis]
  if k >= K:
    mask = jnp.ones(scores.shape, dtype=jnp.bool_)
    return mask if valid_mask is None else mask & valid_mask

  if valid_mask is not None:
    scores = jnp.where(valid_mask, scores, jnp.finfo(jnp.float32).min)
  moved = jnp.moveaxis(scores, axis, -1)
  _, indices = jax.lax.top_k(moved, int(k))
  mask = jax.nn.one_hot(indices, K, dtype=jnp.int32).sum(axis=-2) > 0
  mask = jnp.moveaxis(mask, -1, axis)
  return mask if valid_mask is None else mask & valid_mask


def _straight_through(prob, hard):
  hard = hard.astype(prob.dtype)
  return prob + jax.lax.stop_gradient(hard - prob)


def _masked_mean(values, mask):
  values = values.astype(jnp.float32)
  mask = jnp.broadcast_to(mask, values.shape).astype(jnp.float32)
  denom = jnp.maximum(jnp.sum(mask), 1.0)
  return jnp.sum(values * mask) / denom


def _logit(x, eps=1e-6):
  x = jnp.clip(jnp.asarray(x, dtype=jnp.float32), eps, 1.0 - eps)
  return jnp.log(x) - jnp.log1p(-x)


def _resolve_headwise_kv_heads(cfg: RdmConfig) -> int:
  return cfg.H if cfg.rdm_headwise_kv_heads < 0 else min(
      cfg.rdm_headwise_kv_heads, cfg.H)


def _merge_head_prefix(memory_BxLxHxDh, current_BxLxHxDh, num_memory_heads: int):
  if num_memory_heads >= memory_BxLxHxDh.shape[-2]:
    return memory_BxLxHxDh
  if num_memory_heads <= 0:
    return current_BxLxHxDh
  return jnp.concatenate([
      memory_BxLxHxDh[:, :, :num_memory_heads, :],
      current_BxLxHxDh[:, :, num_memory_heads:, :],
  ], axis=-2)


def _init_edge_mode(cfg: RdmConfig) -> str:
  if cfg.rdm_init_mode != "safe":
    return cfg.rdm_init_mode
  if cfg.rdm_backend == "hc":
    return "closed"
  if cfg.rdm_backend == "dca" and cfg.rdm_block_style == "dca":
    return "dense"
  if cfg.rdm_backend == "dca" and cfg.rdm_source_state == "delta":
    return "dense"
  return "immediate"


def _apply_source_hardening(cfg: RdmConfig, probs_BxLxCxN, valid_1x1x1xN):
  """Apply writer-side hardening over outgoing target edges."""
  if not cfg.rdm_write_hard:
    return probs_BxLxCxN
  if cfg.rdm_hardening == "none":
    return probs_BxLxCxN
  if cfg.rdm_hardening == "st_threshold":
    hard = (probs_BxLxCxN >= cfg.rdm_threshold) & valid_1x1x1xN
    return _straight_through(probs_BxLxCxN, hard)
  if cfg.rdm_hardening == "st_source_topk":
    assert cfg.rdm_k_out > 0, "rdm_k_out must be > 0 for source top-k"
    hard = _one_hot_topk_mask(
        probs_BxLxCxN.astype(jnp.float32), cfg.rdm_k_out, axis=-1,
        valid_mask=valid_1x1x1xN)
    return _straight_through(probs_BxLxCxN, hard)
  # Target top-k is applied after collecting a target inbox.
  return probs_BxLxCxN


def _apply_target_hardening(cfg: RdmConfig, edges_BxLxKxC):
  """Apply incoming-source hardening for the current target layer."""
  if (not cfg.rdm_write_hard) or cfg.rdm_hardening != "st_target_topk":
    return edges_BxLxKxC
  assert cfg.rdm_k_in > 0, "rdm_k_in must be > 0 for target top-k"
  hard = _one_hot_topk_mask(
      edges_BxLxKxC.astype(jnp.float32), cfg.rdm_k_in, axis=2)
  return _straight_through(edges_BxLxKxC, hard)


def sinkhorn_knopp(logits: jax.Array, num_iter: int, eps: float = 1e-6):
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
  """Compute token-wise (H_pre, H_post, H_res) for the HC backend."""

  cfg: RdmConfig

  def _bias_init(self, key, shape, dtype):
    del key, shape
    cfg = self.cfg
    n = cfg.num_streams
    pre = jnp.full((n,), _logit(1.0 / n), dtype=dtype)
    post = jnp.zeros((n,), dtype=dtype)  # 2 * sigmoid(0) = 1.
    if cfg.mhc_h_res_mode == "sinkhorn":
      res = jnp.full((n, n), cfg.rdm_init_closed_logit, dtype=dtype)
      diag = jnp.arange(n)
      res = res.at[diag, diag].set(jnp.asarray(cfg.rdm_init_open_logit,
                                               dtype=dtype))
      return jnp.concatenate([pre, post, res.reshape(-1)], axis=0)
    return jnp.concatenate([pre, post], axis=0)

  @nn.compact
  def __call__(self, streams_BxLxnxD):
    cfg = self.cfg
    n = cfg.num_streams
    B, L, _, D = streams_BxLxnxD.shape
    flat = streams_BxLxnxD.reshape(B, L, n * D)

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
    )(flat)

    rms = jnp.sqrt(jnp.mean(flat.astype(jnp.float32) ** 2, axis=-1, keepdims=True))
    r = (1.0 / (rms + 1e-6)).astype(cfg.dtype)

    init_a = cfg.mhc_init_gating_factor
    alpha_pre = self.param(
        "alpha_pre",
        lambda _: jnp.full((1,), init_a, jnp.dtype(cfg.dtype)),
    )
    alpha_post = self.param(
        "alpha_post",
        lambda _: jnp.full((1,), init_a, jnp.dtype(cfg.dtype)),
    )
    alpha_parts = [
        jnp.broadcast_to(alpha_pre, (n,)),
        jnp.broadcast_to(alpha_post, (n,)),
    ]
    if use_sinkhorn:
      alpha_res = self.param(
          "alpha_res",
          lambda _: jnp.full((1,), init_a, jnp.dtype(cfg.dtype)),
      )
      alpha_parts.append(jnp.broadcast_to(alpha_res, (n * n,)))

    bias_proj = self.param(
        "mapping_bias", self._bias_init, (proj_dim,), jnp.dtype(cfg.dtype)
    )
    alpha = jnp.concatenate(alpha_parts, axis=-1).reshape(1, 1, proj_dim)
    h = r * proj * alpha + bias_proj.reshape(1, 1, proj_dim)

    h_pre = jax.nn.sigmoid(h[..., :n])
    h_post = 2.0 * jax.nn.sigmoid(h[..., n : 2 * n])
    if use_sinkhorn:
      h_res_logits = h[..., 2 * n :].reshape(B, L, n, n)
      h_res = sinkhorn_knopp(h_res_logits, num_iter=cfg.sinkhorn_iters)
    else:
      eye = jnp.eye(n, dtype=jnp.dtype(cfg.dtype))
      h_res = jnp.broadcast_to(eye, (B, L, n, n))
    return h_pre, h_post, h_res.astype(cfg.dtype)


def _mhc_aggregate(streams_BxLxnxD, h_pre_BxLxn):
  return jnp.einsum("blnd,bln->bld", streams_BxLxnxD, h_pre_BxLxn)


def _mhc_step(streams_BxLxnxD, h_post_BxLxn, h_res_BxLxnxn, f_out_BxLxD,
              h_res_is_identity: bool = False):
  expand = jnp.einsum("bln,bld->blnd", h_post_BxLxn, f_out_BxLxD)
  if h_res_is_identity:
    return streams_BxLxnxD + expand
  residual = jnp.einsum("blij,bljd->blid", h_res_BxLxnxn, streams_BxLxnxD)
  return residual + expand


class AttnResReader(nn.Module):
  """Depth-wise softmax reader with a learned pseudo-query per target."""

  cfg: RdmConfig
  layer_index: int
  stream_name: str = "hidden"

  @nn.compact
  def __call__(self, inputs_BxLxKxD, edges_BxLxK=None):
    cfg = self.cfg
    query = self.param(
        "query",
        nn.initializers.zeros,
        (cfg.D,),
        jnp.dtype(cfg.dtype),
    )
    keys = nn.RMSNorm(dtype=cfg.dtype, name="key_norm")(inputs_BxLxKxD)
    base_logits = jnp.einsum("blkd,d->blk", keys, query).astype(jnp.float32)
    logits = base_logits
    if edges_BxLxK is not None and cfg.rdm_writer_edge_apply == "logit_prior":
      logits = logits + cfg.rdm_prior_scale * jnp.log(
          jnp.clip(edges_BxLxK.astype(jnp.float32), cfg.rdm_prior_epsilon, 1.0)
      )
    if edges_BxLxK is not None:
      logits = jnp.where(edges_BxLxK > 0, logits, jnp.finfo(jnp.float32).min)
    alpha = jax.nn.softmax(logits, axis=-1).astype(cfg.dtype)
    if edges_BxLxK is not None:
      alpha = alpha * (edges_BxLxK > 0).astype(alpha.dtype)
      denom = jnp.sum(alpha, axis=-1, keepdims=True)
      alpha = alpha / jnp.maximum(denom, jnp.asarray(cfg.rdm_prior_epsilon,
                                                     dtype=alpha.dtype))
    values = inputs_BxLxKxD
    if edges_BxLxK is not None and cfg.rdm_writer_edge_apply in (
        "value_scale", "inbox_scale"):
      values = values * edges_BxLxK[..., None].astype(values.dtype)
    y = jnp.einsum("blk,blkd->bld", alpha, values)
    alpha0 = jax.nn.softmax(base_logits, axis=-1)
    return y, alpha0.astype(jnp.float32)


class MultiInputRoPECausalAttn(nn.Module):
  """Causal MHA with separate Q/K/V inputs, RoPE, and optional QK RMSNorm."""

  cfg: RdmConfig

  @nn.compact
  def __call__(
      self,
      q_input,
      k_input=None,
      v_input=None,
      positions_BxL=None,
      current_k_input=None,
      current_v_input=None,
  ):
    cfg = self.cfg
    if k_input is None:
      k_input = q_input
    if v_input is None:
      v_input = q_input

    assert cfg.D % cfg.H == 0, f"D {cfg.D} not divisible by H {cfg.H}"
    Dh = cfg.D // cfg.H
    assert Dh % 2 == 0, f"head dim {Dh} must be even for RoPE"
    B, L, _ = q_input.shape
    if positions_BxL is None:
      positions_BxL = jnp.broadcast_to(jnp.arange(L), (B, L))

    multilinear = partial(
        nn.DenseGeneral,
        axis=-1,
        features=(cfg.H, Dh),
        kernel_init=fsdp.init("attn_in_proj", cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )
    query_proj = multilinear(name="query")
    key_proj = multilinear(name="key")
    value_proj = multilinear(name="value")
    q = query_proj(q_input)
    k = key_proj(k_input)
    v = value_proj(v_input)
    km = _resolve_headwise_kv_heads(cfg)
    use_headwise_kv = cfg.rdm_mix_mode in ("qkv", "qkvr") and km < cfg.H
    if use_headwise_kv and (
        current_k_input is None or current_v_input is None):
      raise ValueError(
          "rdm_headwise_kv_heads requires current_k_input/current_v_input")
    if use_headwise_kv:
      k_current = key_proj(current_k_input)
      v_current = value_proj(current_v_input)
    else:
      k_current = None
      v_current = None

    if cfg.qk_norm:
      q = nn.RMSNorm(dtype=cfg.dtype, name="q_norm")(q)
      k_norm = nn.RMSNorm(dtype=cfg.dtype, name="k_norm")
      k = k_norm(k)
      if k_current is not None:
        k_current = k_norm(k_current)

    if cfg.use_rope:
      sin, cos = _rope_sincos(positions_BxL, Dh, cfg.rope_theta, cfg.dtype)
      q = _apply_rope(q, sin, cos)
      k = _apply_rope(k, sin, cos)
      if k_current is not None:
        k_current = _apply_rope(k_current, sin, cos)

    if use_headwise_kv:
      k = _merge_head_prefix(k, k_current, km)
      v = _merge_head_prefix(v, v_current, km)

    q = q / (Dh ** 0.5)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32)
    mask = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(logits, axis=-1).astype(cfg.dtype)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return nn.DenseGeneral(
        features=cfg.D,
        name="attn_out_proj",
        axis=(-2, -1),
        kernel_init=fsdp.init("attn_out_proj", cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )(out)


class MultiInputBlock(nn.Module):
  """Pre-norm Transformer block that can read separate Q/K/V/R streams."""

  cfg: RdmConfig

  @nn.compact
  def __call__(self, xq, xk, xv, xr, positions_BxL=None, current=None):
    cfg = self.cfg
    ln_k = nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="ln_k")
    ln_v = nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="ln_v")
    current_k = None
    current_v = None
    if current is not None and cfg.rdm_mix_mode in ("qkv", "qkvr"):
      current_k = ln_k(current)
      current_v = ln_v(current)
    attn = MultiInputRoPECausalAttn(cfg, name="attn")(
        nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="ln_q")(xq),
        ln_k(xk),
        ln_v(xv),
        positions_BxL,
        current_k,
        current_v,
    )
    if cfg.rdm_block_style == "dca":
      pre_ff = nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="ln_ff")(
          xq + attn)
      z = model.Mlp(cfg, name="mlp")(pre_ff)
      return attn + z
    x = xr + attn
    z = model.Mlp(cfg, name="mlp")(
        nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="ln_ff")(x)
    )
    return x + z


class RoutedWriter(nn.Module):
  """Scheduled source-to-future-target writer."""

  cfg: RdmConfig
  layer_index: int

  def _offset_init(self, key, shape, dtype=jnp.float32):
    del key
    cfg = self.cfg
    mode = _init_edge_mode(cfg)
    init = jnp.full(shape, cfg.rdm_init_closed_logit, dtype=dtype)
    if mode == "dense":
      init = jnp.full(shape, cfg.rdm_init_open_logit, dtype=dtype)
    elif mode == "immediate":
      immediate = self.layer_index + 1
      if immediate < cfg.N:
        init = init.at[:, immediate].set(cfg.rdm_init_open_logit)
    elif mode != "closed":
      raise ValueError(f"unknown resolved init edge mode {mode!r}")
    return init

  @nn.compact
  def __call__(self, hidden_BxLxD, delta_BxLxD):
    cfg = self.cfg
    C = _stream_count(cfg)
    B, L, _ = hidden_BxLxD.shape
    features = jnp.concatenate([
        nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="hidden_ln")(
            hidden_BxLxD),
        nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="delta_ln")(
            delta_BxLxD),
    ], axis=-1)

    if cfg.router_type == "mlp":
      features = nn.Dense(
          2 * cfg.D, use_bias=False, dtype=cfg.dtype,
          kernel_init=cfg.kernel_init, name="router_in")(features)
      features = jax.nn.gelu(features)
    elif cfg.router_type != "linear":
      raise ValueError(f"unknown router_type {cfg.router_type!r}")

    source_logits = nn.Dense(
        C, use_bias=False, dtype=cfg.dtype,
        kernel_init=nn.initializers.zeros,
        name="router_out" if cfg.router_type == "mlp" else "router",
    )(features).astype(jnp.float32)

    if cfg.rdm_writer_version == "simple":
      logits = jnp.broadcast_to(source_logits[..., None], (B, L, C, cfg.N))
    elif cfg.rdm_writer_version == "targeted":
      rank = cfg.rdm_writer_rank
      source_proj = nn.Dense(
          C * rank, use_bias=False, dtype=cfg.dtype,
          kernel_init=nn.initializers.zeros, name="source_proj")(features)
      source_proj = source_proj.reshape(B, L, C, rank).astype(jnp.float32)
      target_emb = self.param(
          "target_emb",
          nn.initializers.normal(stddev=0.02),
          (cfg.N, C, rank),
      ).astype(jnp.float32)
      compat = jnp.einsum("blcr,ncr->blcn", source_proj, target_emb)
      offset = self.param("target_bias", self._offset_init, (C, cfg.N))
      logits = source_logits[..., None] + compat + offset[None, None, :, :]
    else:
      raise ValueError(f"unknown rdm_writer_version {cfg.rdm_writer_version!r}")

    valid = (jnp.arange(cfg.N) > self.layer_index).reshape(1, 1, 1, cfg.N)
    logits = jnp.where(valid, logits, jnp.finfo(jnp.float32).min)
    probs = jax.nn.sigmoid(logits / cfg.rdm_temperature).astype(cfg.dtype)
    probs = jnp.where(valid, probs, jnp.zeros_like(probs))
    probs = _apply_source_hardening(cfg, probs, valid)

    if cfg.rdm_z_weight > 0.0:
      valid_logits = jnp.where(valid, logits, jnp.finfo(jnp.float32).min)
      z = jnp.mean(jax.nn.logsumexp(valid_logits, axis=-1) ** 2)
      self.sow(
          "aux_loss", "rdm_zloss", cfg.rdm_z_weight * z,
          reduce_fn=lambda a, b: a + b, init_fn=lambda: jnp.asarray(0.0))
    return probs


class RoutedInputSelector(nn.Module):
  """DCA `InputSelector` with optional writer prior/mask support."""

  cfg: RdmConfig
  stream_name: str = "hidden"

  @nn.compact
  def __call__(self, inputs_BxLxKxD, edges_BxLxK=None):
    cfg = self.cfg
    Dim = cfg.D
    K = inputs_BxLxKxD.shape[-2]
    partition_fn = (
        nn.with_partitioning if cfg.fsdp_enabled else lambda x, y: x
    )
    query = self.param(
        "query",
        partition_fn(nn.initializers.zeros, ("data",)),
        (Dim,),
    )
    bias = self.param(
        "bias",
        partition_fn(nn.initializers.ones, (None, "data")),
        (K, Dim),
    )

    keys = nn.LayerNorm(dtype=cfg.dtype)(inputs_BxLxKxD)
    base_logits = jnp.einsum("...li,i->...l", keys, query).astype(jnp.float32)
    logits = base_logits

    if edges_BxLxK is not None and cfg.rdm_writer_edge_apply == "logit_prior":
      prior = jnp.log(
          jnp.clip(edges_BxLxK.astype(jnp.float32),
                   cfg.rdm_prior_epsilon, 1.0))
      logits = logits + cfg.rdm_prior_scale * prior

    coeff = nn.relu(logits[..., None] + bias[None, None, :, :])
    keep = dca_ref._grn_keep_mask(coeff.mean(axis=-1), cfg)
    if keep is not None:
      coeff = coeff * keep[..., None]

    if edges_BxLxK is not None:
      active = edges_BxLxK > 0
      # Mask the full bias-inclusive coefficient so closed edges cannot leak.
      coeff = coeff * active[..., None]
      if cfg.rdm_writer_edge_apply in ("value_scale", "inbox_scale"):
        coeff = coeff * edges_BxLxK[..., None].astype(coeff.dtype)

    y = jnp.einsum("...li,...li->...i", inputs_BxLxKxD, coeff)

    probe_coeff = nn.relu(base_logits[..., None] + bias[None, None, :, :])
    probe_scores = probe_coeff.mean(axis=-1)
    denom = jnp.sum(probe_scores, axis=-1, keepdims=True)
    alpha0 = probe_scores / jnp.maximum(denom, cfg.rdm_prior_epsilon)
    return y, alpha0.astype(jnp.float32)


def _edge_for_stream(edges_BxLxKxC, stream_index: int):
  if edges_BxLxKxC.shape[-1] == 1:
    return edges_BxLxKxC[..., 0]
  return edges_BxLxKxC[..., stream_index]


def _hc_inbox_edges(cfg: RdmConfig, edges_BxLxKxC):
  if cfg.rdm_writer_edge_apply == "mask_only":
    return (edges_BxLxKxC > 0).astype(edges_BxLxKxC.dtype)
  return edges_BxLxKxC


class RdmTransformerDo(nn.Module):
  """Decoder-only Transformer with scheduled depth routing."""

  docfg: RdmConfig

  def _sow_intermediate(self, name, value):
    self.sow(
        "intermediate_acts",
        name,
        jnp.asarray(value, dtype=jnp.float32),
        reduce_fn=lambda a, b: a + b,
        init_fn=lambda: jnp.asarray(0.0, dtype=jnp.float32),
    )

  def _reader_calls_per_forward(self):
    cfg = self.docfg
    if cfg.rdm_mix_mode == "hidden":
      return max(cfg.N, 1)
    if cfg.rdm_mix_mode == "qkv":
      return max(3 * cfg.N, 1)
    return max(4 * cfg.N, 1)

  def _sow_reader_metrics(self, alpha0_BxLxK, edges_BxLxK):
    cfg = self.docfg
    weight = 1.0 / self._reader_calls_per_forward()
    K = alpha0_BxLxK.shape[-1]
    clipped = jnp.clip(alpha0_BxLxK, cfg.rdm_prior_epsilon, 1.0)
    entropy = -jnp.sum(alpha0_BxLxK * jnp.log(clipped), axis=-1)
    if K > 1:
      entropy = entropy / jnp.log(jnp.asarray(K, dtype=jnp.float32))
    else:
      entropy = jnp.zeros_like(entropy)
    self._sow_intermediate("rdm_reader_alpha_entropy", weight * jnp.mean(entropy))
    self._sow_intermediate(
        "rdm_reader_alpha_max",
        weight * jnp.mean(jnp.max(alpha0_BxLxK, axis=-1)),
    )
    self._sow_intermediate(
        "rdm_reader_edge_alignment",
        weight * jnp.mean(jnp.sum(alpha0_BxLxK * edges_BxLxK, axis=-1)),
    )

  def _sow_target_edge_metrics(self, edges_BxLxKxC):
    cfg = self.docfg
    weight = 1.0 / max(cfg.N, 1)
    edges = edges_BxLxKxC.astype(jnp.float32)
    hard = edges >= cfg.rdm_threshold
    self._sow_intermediate("rdm_target_edge_mean", weight * jnp.mean(edges))
    self._sow_intermediate(
        "rdm_target_edge_ge_threshold", weight * jnp.mean(hard.astype(jnp.float32))
    )
    self._sow_intermediate(
        "rdm_target_incoming_soft_count",
        weight * jnp.mean(jnp.sum(edges, axis=2)),
    )
    self._sow_intermediate(
        "rdm_target_incoming_hard_count",
        weight * jnp.mean(jnp.sum(hard.astype(jnp.float32), axis=2)),
    )
    if edges.shape[-1] > 1:
      per_stream = jnp.mean(jnp.sum(edges, axis=2), axis=(0, 1))
      probs = per_stream / jnp.maximum(jnp.sum(per_stream), cfg.rdm_prior_epsilon)
      entropy = -jnp.sum(probs * jnp.log(jnp.clip(probs, cfg.rdm_prior_epsilon, 1.0)))
      entropy = entropy / jnp.log(jnp.asarray(edges.shape[-1], dtype=jnp.float32))
      self._sow_intermediate("rdm_target_stream_entropy", weight * entropy)

  def _sow_writer_metrics(self, edge_row_BxLxCxN, layer_index: int):
    cfg = self.docfg
    weight = 1.0 / max(cfg.N - 1, 1)
    valid = (jnp.arange(cfg.N) > layer_index).reshape(1, 1, 1, cfg.N)
    long_range = (jnp.arange(cfg.N) > layer_index + 1).reshape(1, 1, 1, cfg.N)
    edges = edge_row_BxLxCxN.astype(jnp.float32)
    hard = edges >= cfg.rdm_threshold
    valid_edges = jnp.where(valid, edges, jnp.zeros_like(edges))
    valid_hard = jnp.where(valid, hard, jnp.zeros_like(hard))

    self._sow_intermediate("rdm_writer_edge_mean", weight * _masked_mean(edges, valid))
    self._sow_intermediate(
        "rdm_writer_edge_ge_threshold",
        weight * _masked_mean(hard.astype(jnp.float32), valid),
    )
    self._sow_intermediate(
        "rdm_writer_outgoing_soft_count",
        weight * jnp.mean(jnp.sum(valid_edges, axis=-1)),
    )
    self._sow_intermediate(
        "rdm_writer_outgoing_hard_count",
        weight * jnp.mean(jnp.sum(valid_hard.astype(jnp.float32), axis=-1)),
    )
    if layer_index + 1 < cfg.N:
      immediate = edges[..., layer_index + 1]
      self._sow_intermediate(
          "rdm_writer_immediate_edge_mean", weight * jnp.mean(immediate)
      )
    self._sow_intermediate(
        "rdm_writer_long_range_edge_mean",
        weight * _masked_mean(edges, long_range),
    )
    if edges.shape[-2] > 1:
      per_stream = jnp.mean(jnp.sum(valid_edges, axis=-1), axis=(0, 1))
      probs = per_stream / jnp.maximum(jnp.sum(per_stream), cfg.rdm_prior_epsilon)
      entropy = -jnp.sum(probs * jnp.log(jnp.clip(probs, cfg.rdm_prior_epsilon, 1.0)))
      entropy = entropy / jnp.log(jnp.asarray(edges.shape[-2], dtype=jnp.float32))
      self._sow_intermediate("rdm_writer_stream_entropy", weight * entropy)

  def _initial_embedding_edges(self, cfg, B, L, C):
    if not cfg.rdm_writer_enabled:
      return jnp.ones((B, L, C, cfg.N), dtype=cfg.dtype)
    mode = _init_edge_mode(cfg)
    if cfg.rdm_writer_edge_apply == "mask_only":
      row = jnp.zeros((B, L, C, cfg.N), dtype=cfg.dtype)
      if mode == "dense":
        return jnp.ones((B, L, C, cfg.N), dtype=cfg.dtype)
      if mode == "immediate" and cfg.N > 0:
        row = row.at[:, :, :, 0].set(jnp.ones((), dtype=cfg.dtype))
      elif mode != "closed":
        raise ValueError(f"unknown resolved init edge mode {mode!r}")
      return row
    closed = jax.nn.sigmoid(
        jnp.asarray(cfg.rdm_init_closed_logit / cfg.rdm_temperature,
                    dtype=jnp.float32))
    open_ = jax.nn.sigmoid(
        jnp.asarray(cfg.rdm_init_open_logit / cfg.rdm_temperature,
                    dtype=jnp.float32))
    row = jnp.full((B, L, C, cfg.N), closed, dtype=cfg.dtype)
    if mode == "dense":
      row = jnp.full((B, L, C, cfg.N), open_, dtype=cfg.dtype)
    elif mode == "immediate" and cfg.N > 0:
      row = row.at[:, :, :, 0].set(open_.astype(cfg.dtype))
    elif mode != "closed":
      raise ValueError(f"unknown resolved init edge mode {mode!r}")
    return row

  def _sow_align(self, alpha0_BxLxK, edges_BxLxK):
    cfg = self.docfg
    if cfg.rdm_align_weight <= 0.0:
      return
    log_edges = jnp.log(
        jnp.clip(edges_BxLxK.astype(jnp.float32), cfg.rdm_prior_epsilon, 1.0))
    loss = -jnp.mean(jax.lax.stop_gradient(alpha0_BxLxK) * log_edges)
    self.sow(
        "aux_loss", "rdm_align", cfg.rdm_align_weight * loss,
        reduce_fn=lambda a, b: a + b, init_fn=lambda: jnp.asarray(0.0))

  def _read_stream(
      self, layer_index, name, stack_BxLxKxD, edges_BxLxK, stream_index):
    y, alpha0 = RoutedInputSelector(
        self.docfg, stream_name=name, name=f"reader_{layer_index}_{name}")(
            stack_BxLxKxD, edges_BxLxK)
    del stream_index
    self._sow_align(alpha0, edges_BxLxK)
    self._sow_reader_metrics(alpha0, edges_BxLxK)
    return y

  def _read_attnres_stream(
      self, layer_index, name, stack_BxLxKxD, edges_BxLxK, stream_index):
    y, alpha0 = AttnResReader(
        self.docfg, layer_index=layer_index, stream_name=name,
        name=f"attnres_{layer_index}_{name}")(
            stack_BxLxKxD, edges_BxLxK)
    del stream_index
    self._sow_align(alpha0, edges_BxLxK)
    self._sow_reader_metrics(alpha0, edges_BxLxK)
    return y

  def _source_message(self, h_next, h_prev, delta, layer_index):
    cfg = self.docfg
    if cfg.rdm_source_state == "hidden":
      return h_next
    if cfg.rdm_source_state == "delta":
      return delta
    if cfg.rdm_source_state == "ln_hidden":
      return nn.LayerNorm(
          dtype=cfg.dtype, use_bias=False, name=f"write_ln_hidden_{layer_index}")(
              h_next)
    del h_prev
    return nn.LayerNorm(
        dtype=cfg.dtype, use_bias=False, name=f"write_ln_delta_{layer_index}")(
            delta)

  def _append_writer_row(self, edge_rows, h_next, delta, layer_index, B, L, C):
    cfg = self.docfg
    if cfg.rdm_writer_enabled:
      edge_row = RoutedWriter(cfg, layer_index=layer_index,
                              name=f"writer_{layer_index}")(h_next, delta)
    else:
      edge_row = jnp.ones((B, L, C, cfg.N), dtype=cfg.dtype)
    self._sow_writer_metrics(edge_row, layer_index)
    return edge_rows + [edge_row]

  def _read_edges_for_layer(self, edge_rows, layer_index, B, L, K, C):
    cfg = self.docfg
    if edge_rows and cfg.rdm_writer_enabled:
      edges = jnp.stack([row[:, :, :, layer_index] for row in edge_rows], axis=2)
    else:
      edges = jnp.ones((B, L, K, C), dtype=cfg.dtype)
    edges = _apply_target_hardening(cfg, edges)
    self._sow_target_edge_metrics(edges)
    return edges

  def _block_groups(self, K: int):
    cfg = self.docfg
    if cfg.attnres_variant == "full":
      return tuple((i,) for i in range(K))
    block = cfg.attnres_block_size
    return tuple(tuple(range(start, min(start + block, K)))
                 for start in range(0, K, block))

  def _group_attnres_inputs(self, stack_BxLxKxD, edges_BxLxKxC):
    groups = self._block_groups(stack_BxLxKxD.shape[2])
    if len(groups) == stack_BxLxKxD.shape[2]:
      return stack_BxLxKxD, edges_BxLxKxC
    summaries = []
    edge_summaries = []
    for group in groups:
      summaries.append(jnp.sum(stack_BxLxKxD[:, :, group, :], axis=2))
      edge_summaries.append(jnp.max(edges_BxLxKxC[:, :, group, :], axis=2))
    return jnp.stack(summaries, axis=2), jnp.stack(edge_summaries, axis=2)

  def _call_depth_reader(self, layer_index, stack, edges, reader_kind):
    read = self._read_stream if reader_kind == "dca" else self._read_attnres_stream
    cfg = self.docfg
    if cfg.rdm_mix_mode == "hidden":
      e = _edge_for_stream(edges, 0)
      x = read(layer_index, "hidden", stack, e, 0)
      return x, x, x, x
    if cfg.rdm_mix_mode == "qkv":
      xq = read(layer_index, "q", stack, _edge_for_stream(edges, 0), 0)
      xk = read(layer_index, "k", stack, _edge_for_stream(edges, 1), 1)
      xv = read(layer_index, "v", stack, _edge_for_stream(edges, 2), 2)
      return xq, xk, xv, None
    streams = [
        read(layer_index, stream, stack, _edge_for_stream(edges, idx), idx)
        for idx, stream in enumerate(_STREAMS_QKVR)
    ]
    return tuple(streams)

  def _call_dca(self, h, positions, B, L, C):
    cfg = self.docfg
    messages = []
    edge_rows = []
    if cfg.rdm_include_embedding:
      messages.append(h)
      edge_rows.append(self._initial_embedding_edges(cfg, B, L, C))

    for lyr in range(cfg.N):
      read_messages = messages
      read_edges = edge_rows
      if (not cfg.rdm_include_immediate_prev) and len(read_messages) > 1:
        read_messages = read_messages[:-1]
        read_edges = read_edges[:-1]

      if read_messages:
        stack = jnp.stack(read_messages, axis=2)
        edges = self._read_edges_for_layer(
            read_edges, lyr, B, L, stack.shape[2], C)
      else:
        stack = h[:, :, None, :]
        edges = self._read_edges_for_layer([], lyr, B, L, 1, C)

      xq, xk, xv, xr = self._call_depth_reader(lyr, stack, edges, "dca")
      if xr is None:
        xr = h
      h_next = MultiInputBlock(cfg, name=f"block_{lyr}")(
          xq, xk, xv, xr, positions, current=h)
      delta = h_next if cfg.rdm_block_style == "dca" else h_next - h

      if lyr + 1 < cfg.N:
        messages.append(self._source_message(h_next, h, delta, lyr))
        edge_rows = self._append_writer_row(edge_rows, h_next, delta, lyr,
                                            B, L, C)
      h = h_next
    return h

  def _call_hc(self, h, positions, B, L, C):
    cfg = self.docfg
    n = cfg.num_streams
    streams = jnp.broadcast_to(h[:, :, None, :], (B, L, n, cfg.D))
    streams = jnp.asarray(streams)
    messages = []
    edge_rows = []
    if cfg.rdm_include_embedding:
      messages.append(h)
      emb_edges = self._initial_embedding_edges(cfg, B, L, C)
      # HC streams already start from the token embedding; do not inject that
      # same embedding inbox again into target layer 0.
      emb_edges = emb_edges.at[:, :, :, 0].set(jnp.zeros((), dtype=cfg.dtype))
      edge_rows.append(emb_edges)

    is_identity = cfg.mhc_h_res_mode == "identity"
    for lyr in range(cfg.N):
      prev_hidden = streams.mean(axis=2)

      # Scheduled writer inbox: inject visible source messages into the stream
      # state before the standard mHC attention/MLP transitions.
      if messages:
        read_messages = messages
        read_edges = edge_rows
        if (not cfg.rdm_include_immediate_prev) and len(read_messages) > 1:
          read_messages = read_messages[:-1]
          read_edges = read_edges[:-1]
        stack = jnp.stack(read_messages, axis=2)
        edges = self._read_edges_for_layer(
            read_edges, lyr, B, L, stack.shape[2], C)
        inbox_edges = _hc_inbox_edges(cfg, edges)
        h_pre, h_post, h_res = MHCMappings(cfg, name=f"map_inbox_{lyr}")(streams)
        del h_pre
        if C == 1:
          inbox_base = jnp.einsum("blk,blkd->bld", inbox_edges[..., 0], stack)
          inbox = jnp.einsum("bln,bld->blnd", h_post, inbox_base)
        else:
          inbox = jnp.einsum("blkn,blkd->blnd", inbox_edges, stack)
        if is_identity:
          streams = streams + inbox
        else:
          streams = jnp.einsum("blij,bljd->blid", h_res, streams) + inbox
      else:
        self._sow_target_edge_metrics(jnp.ones((B, L, 1, C), dtype=cfg.dtype))

      h_pre, h_post, h_res = MHCMappings(cfg, name=f"map_attn_{lyr}")(streams)
      agg = _mhc_aggregate(streams, h_pre)
      attn = MultiInputRoPECausalAttn(cfg, name=f"attn_{lyr}")(
          nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name=f"ln_attn_{lyr}")(
              agg),
          positions_BxL=positions,
      )
      streams = _mhc_step(streams, h_post, h_res, attn, is_identity)

      h_pre, h_post, h_res = MHCMappings(cfg, name=f"map_mlp_{lyr}")(streams)
      agg = _mhc_aggregate(streams, h_pre)
      mlp_out = model.Mlp(cfg, name=f"mlp_{lyr}")(
          nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name=f"ln_mlp_{lyr}")(
              agg)
      )
      streams = _mhc_step(streams, h_post, h_res, mlp_out, is_identity)
      h_next = streams.mean(axis=2)
      delta = h_next - prev_hidden

      if lyr + 1 < cfg.N:
        messages.append(self._source_message(h_next, prev_hidden, delta, lyr))
        edge_rows = self._append_writer_row(edge_rows, h_next, delta, lyr,
                                            B, L, C)
    return streams.mean(axis=2)

  def _call_attnres(self, h, positions, B, L, C):
    cfg = self.docfg
    messages = []
    edge_rows = []
    if cfg.rdm_include_embedding:
      messages.append(h)
      edge_rows.append(self._initial_embedding_edges(cfg, B, L, C))

    for lyr in range(cfg.N):
      read_messages = messages
      read_edges = edge_rows
      if (not cfg.rdm_include_immediate_prev) and len(read_messages) > 1:
        read_messages = read_messages[:-1]
        read_edges = read_edges[:-1]
      if read_messages:
        stack = jnp.stack(read_messages, axis=2)
        edges = self._read_edges_for_layer(
            read_edges, lyr, B, L, stack.shape[2], C)
        stack, edges = self._group_attnres_inputs(stack, edges)
      else:
        stack = h[:, :, None, :]
        edges = self._read_edges_for_layer([], lyr, B, L, 1, C)

      xq, xk, xv, xr = self._call_depth_reader(lyr, stack, edges, "attnres")
      if xr is None:
        xr = h
      h_next = MultiInputBlock(cfg, name=f"block_{lyr}")(
          xq, xk, xv, xr, positions, current=h)
      delta = h_next - h

      if lyr + 1 < cfg.N:
        messages.append(self._source_message(h_next, h, delta, lyr))
        edge_rows = self._append_writer_row(edge_rows, h_next, delta, lyr,
                                            B, L, C)
      h = h_next
    return h

  @nn.compact
  def __call__(self, y_BxL):
    cfg = self.docfg
    _validate_cfg(cfg)
    B, L = y_BxL.shape
    C = _stream_count(cfg)
    positions = jnp.broadcast_to(jnp.arange(L), (B, L))

    embed = nn.Embed(
        cfg.V, cfg.D, embedding_init=fsdp.init("embedding", cfg), name="embed")
    h = embed(y_BxL)
    if not cfg.use_rope:
      pos_embed = nn.Embed(
          cfg.L, cfg.D, embedding_init=fsdp.init("embedding", cfg),
          name="pos_embed")
      h = h + pos_embed(jnp.arange(0, L)[None, ...])

    if cfg.rdm_backend == "dca":
      h = self._call_dca(h, positions, B, L, C)
    elif cfg.rdm_backend == "hc":
      h = self._call_hc(h, positions, B, L, C)
    else:
      h = self._call_attnres(h, positions, B, L, C)

    h = nn.LayerNorm(dtype=cfg.dtype, use_bias=False, name="out_ln")(h)
    return embed.attend(h.astype(jnp.float32))
