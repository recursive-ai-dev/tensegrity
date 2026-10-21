"""TensegrityLM: a byte-level decoder-only language model.

Mechanisms (each is established in the literature; the *combination* is a
coherent but unvalidated recombination -- measure before claiming it beats a
vanilla transformer of equal size):

  * RoPE positions                        * RMSNorm pre-norm, optional Q/K RMSNorm
  * Grouped-query attention               * alternating local (banded) / global attention
  * SwiGLU feed-forward                   * learnable per-layer residual gains
  * activation-RMS "homeostasis" loss     * depth-scaled LayerDrop
  * z-loss, logit soft-capping            * optional confidence head (adaptive sampling temperature)
  * optional ByteMix (causal FIR over recent byte embeddings) and x0-mix (embedding re-injection)
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import TensegrityConfig
from .layers import (Block, ByteMix, KV, RMSNorm, apply_logit_soft_cap, precompute_rope)
from .tokenizer import ByteTokenizer


class TensegrityLM(nn.Module):
    def __init__(self, cfg: TensegrityConfig, verbose: bool = False):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpoint = cfg.grad_checkpoint
        self.grad_checkpoint_every = cfg.grad_checkpoint_every

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.byte_mix = None
        if cfg.byte_mix_kernel:
            if cfg.vocab_size <= ByteTokenizer.PAD:
                raise ValueError("byte_mix_kernel needs vocab_size > PAD id (258)")
            self.byte_mix = ByteMix(cfg.byte_mix_kernel, cfg.n_embd,
                                    bilinear=cfg.byte_mix_bilinear, init_std=cfg.init_std,
                                    embed_scale=cfg.embed_scale)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying

        self.confidence_head = nn.Linear(cfg.n_embd, 1, bias=True) if cfg.use_confidence_head else None

        # RoPE tables live outside the module state (not buffers): they stay fp32 under
        # `.half()` and never appear in a state_dict. Built lazily per device.
        self._rope: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # LayerDrop draws come from a private CPU generator: no GPU sync, no collective,
        # and identical on every DDP rank as long as every rank seeds it identically.
        self._ld_gen = torch.Generator(device="cpu")
        self._ld_gen.manual_seed(0)
        self._drop_ps: List[float] = [b.drop_p for b in self.blocks]

        self.apply(self._init_weights)
        if verbose:
            print(f"[TensegrityLM] {cfg.describe()}")

    # ------------------------------------------------------------------ setup
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.token_emb.weight.numel() if non_embedding else n

    def summary(self) -> str:
        return f"{self.cfg.describe()} | counted {self.num_parameters():,} params"

    def _rope_tables(self, device: torch.device, n: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for at least ``max(block_size, n)`` positions (streaming decode asks for more)."""
        key = str(device)
        need = max(self.cfg.block_size, n or 0)
        tabs = self._rope.get(key)
        if tabs is None or tabs[0].shape[0] < need:
            grown = need if tabs is None else max(need, 2 * tabs[0].shape[0])
            tabs = precompute_rope(self.cfg.head_dim, grown, self.cfg.rope_base, device=device)
            self._rope[key] = tabs
        return tabs

    # ------------------------------------------------------------- layerdrop
    def seed_layerdrop(self, seed: int) -> None:
        self._ld_gen.manual_seed(int(seed))

    def layerdrop_state(self) -> torch.Tensor:
        return self._ld_gen.get_state()

    def set_layerdrop_state(self, state: torch.Tensor) -> None:
        self._ld_gen.set_state(state.cpu())

    def _layerdrop_plan(self) -> List[bool]:
        n = len(self.blocks)
        if not self.training or not any(p > 0.0 for p in self._drop_ps):
            return [False] * n
        u = torch.rand(n, generator=self._ld_gen).tolist()
        return [u[i] < self._drop_ps[i] for i in range(n)]

    # --------------------------------------------------------------- embedding
    def _embed(self, idx: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Token ids -> embedding stream x0, shape (B, T, C).

        ``ctx`` holds the (up to K-1) tokens that immediately precede ``idx`` (used only by
        ByteMix during cached decoding); missing left context is filled with PAD, which is
        exactly what training sees at the start of every window.
        """
        cfg = self.cfg
        if self.byte_mix is None:
            x = self.token_emb(idx)
            if cfg.embed_scale:
                x = x * (cfg.n_embd ** 0.5)
            return self.drop(x)

        n_prefix = cfg.byte_mix_kernel - 1
        B = idx.size(0)
        if ctx is None or ctx.numel() == 0:
            prefix = idx.new_full((B, n_prefix), ByteTokenizer.PAD)
        else:
            prefix = ctx[:, -n_prefix:]
            short = n_prefix - prefix.size(1)
            if short > 0:
                prefix = torch.cat([idx.new_full((B, short), ByteTokenizer.PAD), prefix], dim=1)
        e = self.token_emb(torch.cat([prefix, idx], dim=1))
        if cfg.embed_scale:
            e = e * (cfg.n_embd ** 0.5)
        return self.drop(self.byte_mix(e, n_prefix))

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(self.ln_f(x))
        if self.cfg.logit_soft_cap is not None:
            logits = apply_logit_soft_cap(logits, self.cfg.logit_soft_cap)
        return logits

    # ----------------------------------------------------------------- forward
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """
        idx:     (B, T) token ids, T <= block_size
        targets: (B, T) next-token ids (-1 = ignore) or None

        Returns ``(logits, losses, confidence)``:
          * with targets: ``losses`` is a dict (ce_loss, homeo_loss, total_loss, [z_loss],
            [confidence_loss]) and ``confidence`` is None;
          * without: ``losses`` is None and ``confidence`` is (B, T) in [0, 1] (or None).
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")

        x0 = self._embed(idx)
        x = x0
        x0_arg = x0 if self.cfg.x0_mix else None
        cos_all, sin_all = self._rope_tables(idx.device)
        cos = cos_all[:T].view(1, 1, T, -1)
        sin = sin_all[:T].view(1, 1, T, -1)

        plan = self._layerdrop_plan()
        use_ckpt = self.grad_checkpoint and self.training and x.requires_grad
        total_homeo = x.new_zeros((), dtype=torch.float32)
        n_active = 0
        v0: Optional[torch.Tensor] = None
        need_v0_flag = self.cfg.v0_mix
        for loop_idx in range(self.cfg.n_loops):
            for i, block in enumerate(self.blocks):
                if plan[i]:
                    continue
                need_v0 = need_v0_flag and v0 is None
                if use_ckpt and i % self.grad_checkpoint_every == 0:
                    x, homeo, v0 = torch.utils.checkpoint.checkpoint(
                        block._forward_impl, x, cos, sin, x0_arg, v0, loop_idx, need_v0, use_reentrant=False)
                else:
                    x, homeo, v0 = block._forward_impl(x, cos, sin, x0_arg, v0, loop_idx, need_v0)
                total_homeo = total_homeo + homeo
                n_active += 1

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if self.cfg.logit_soft_cap is not None:
            logits = apply_logit_soft_cap(logits, self.cfg.logit_soft_cap)

        if targets is None:
            confidence = None
            if self.confidence_head is not None:
                confidence = torch.sigmoid(self.confidence_head(x).squeeze(-1))
            return logits, None, confidence

        cfg = self.cfg
        logits_f = logits.float()
        ce_loss = F.cross_entropy(logits_f.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        avg_homeo = total_homeo / max(1, n_active * 2)
        total = ce_loss + cfg.homeo_weight * avg_homeo
        losses: Dict[str, torch.Tensor] = {"ce_loss": ce_loss, "homeo_loss": avg_homeo}

        if cfg.z_loss_weight > 0.0:
            z = cfg.z_loss_weight * torch.logsumexp(logits_f, dim=-1).pow(2).mean()
            losses["z_loss"] = z
            total = total + z

        if self.confidence_head is not None:
            h = x.detach() if cfg.confidence_detach else x
            conf_logits = self.confidence_head(h).squeeze(-1)
            with torch.no_grad():
                correct = (logits.argmax(dim=-1) == targets).float()
                valid = (targets != -1).float()
            raw = F.binary_cross_entropy_with_logits(conf_logits.float(), correct, reduction="none")
            conf_loss = (raw * valid).sum() / valid.sum().clamp_min(1.0)
            losses["confidence_loss"] = conf_loss
            total = total + cfg.confidence_weight * conf_loss

        losses["total_loss"] = total
        return logits, losses, None

    # ------------------------------------------------------ cached incremental
    def forward_incremental(self, idx: torch.Tensor,
                            past_kv_list: Optional[List[Optional[KV]]] = None,
                            pos_start: int = 0,
                            ctx: Optional[torch.Tensor] = None,
                            streaming: bool = False,
                            ) -> Tuple[torch.Tensor, List[KV], Optional[torch.Tensor]]:
        """KV-cached forward for inference.

        * prefill: ``past_kv_list=None``, ``idx`` is a whole window starting at position 0.
        * decode:  ``past_kv_list`` from the previous call, ``idx`` is ONE new token at
          absolute window position ``pos_start``; ``ctx`` = the K-1 tokens before it
          (needed only when ``byte_mix_kernel`` is set).
        * ``streaming=True``: ``pos_start`` may run past ``block_size`` (absolute stream position); global
          layers keep only their most recent ``block_size`` keys, so memory and per-token cost stay bounded.
          RoPE is relative, so every retained key stays within the trained distance range of the query.

        Returns (logits, new_past_kv_list, confidence) for the T new positions.
        """
        B, T = idx.shape
        if not streaming and pos_start + T > self.cfg.block_size:
            raise ValueError(f"positions [{pos_start}, {pos_start + T}) exceed block_size {self.cfg.block_size}")
        if streaming and T > self.cfg.block_size:
            raise ValueError(f"a streaming step takes at most block_size={self.cfg.block_size} tokens, got {T}")
        max_cache = self.cfg.block_size if streaming else None
        n_total = len(self.blocks) * self.cfg.n_loops      # one KV cache per (loop pass, block)
        if past_kv_list is None:
            past_kv_list = [None] * n_total

        x0 = self._embed(idx, ctx)
        x = x0
        x0_arg = x0 if self.cfg.x0_mix else None
        cos_all, sin_all = self._rope_tables(idx.device, pos_start + T)
        cos = cos_all[pos_start:pos_start + T].view(1, 1, T, -1)
        sin = sin_all[pos_start:pos_start + T].view(1, 1, T, -1)

        new_past: List[KV] = []
        v0: Optional[torch.Tensor] = None
        need_v0_flag = self.cfg.v0_mix
        kv_idx = 0
        for loop_idx in range(self.cfg.n_loops):
            for block in self.blocks:
                past = past_kv_list[kv_idx]
                kv_idx += 1
                if need_v0_flag and v0 is None:
                    x, kv, v0 = block.forward_incremental(x, cos, sin, x0_arg, past, max_cache, v0=None,
                                                          loop_idx=loop_idx, return_v0=True)
                else:
                    x, kv = block.forward_incremental(x, cos, sin, x0_arg, past, max_cache, v0=v0,
                                                      loop_idx=loop_idx, return_v0=False)
                new_past.append(kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if self.cfg.logit_soft_cap is not None:
            logits = apply_logit_soft_cap(logits, self.cfg.logit_soft_cap)
        confidence = None
        if self.confidence_head is not None:
            confidence = torch.sigmoid(self.confidence_head(x).squeeze(-1))
        return logits, new_past, confidence

    # --------------------------------------------------------------- sampling
    @staticmethod
    def _sample_next(logits: torch.Tensor, temperature: float, top_k: Optional[int],
                     top_p: Optional[float], min_p: Optional[float],
                     conf: Optional[torch.Tensor], conf_temp_min: float, conf_temp_max: float
                     ) -> torch.Tensor:
        """logits: (B, V) -> next token ids (B, 1)."""
        logits = logits.float()
        if temperature == 0.0:  # an explicit greedy request always wins over confidence scaling
            return logits.argmax(dim=-1, keepdim=True)

        if conf is not None:
            # low confidence -> hotter, high confidence -> cooler (per sequence)
            scale = conf_temp_max - conf.float() * (conf_temp_max - conf_temp_min)
            logits = logits / (temperature * scale).clamp_min(1e-4).unsqueeze(-1)
        else:
            logits = logits / max(temperature, 1e-4)
        fallback = logits.argmax(dim=-1, keepdim=True)

        if top_k is not None:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            cut = cum > top_p
            cut[:, 1:] = cut[:, :-1].clone()  # keep the first token that crosses top_p
            cut[:, 0] = False
            logits = torch.full_like(logits, float("-inf")).scatter(
                1, sorted_idx, sorted_logits.masked_fill(cut, float("-inf")))
        if min_p is not None:
            p = F.softmax(logits, dim=-1)
            logits = logits.masked_fill(p < min_p * p.max(dim=-1, keepdim=True).values, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        bad = ~torch.isfinite(probs).all(dim=-1) | (probs.sum(dim=-1) <= 0.0)
        if bad.any():  # degenerate row (all -inf / NaN): fall back to greedy for that row only
            probs = torch.where(bad.unsqueeze(-1), torch.ones_like(probs), torch.nan_to_num(probs, nan=0.0))
            sampled = torch.multinomial(probs, 1)
            return torch.where(bad.unsqueeze(-1), fallback, sampled)
        return torch.multinomial(probs, 1)

    def _generate_stream_iter(self, idx: torch.Tensor, max_new_tokens: int, temperature: float,
                              top_k: Optional[int], top_p: Optional[float], min_p: Optional[float],
                              use_confidence_temp: bool, conf_temp_min: float, conf_temp_max: float,
                              stop_on_eos: bool) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Streaming-cache decoding: one prefill, then O(1) work per token forever (see ``generate``)."""
        bs, K = self.cfg.block_size, self.cfg.byte_mix_kernel
        m = max(K - 1, 0)                                   # ByteMix needs the previous K-1 tokens
        was_training = self.training
        self.eval()
        try:
            B = idx.size(0)
            finished = torch.zeros(B, dtype=torch.bool, device=idx.device) if stop_on_eos else None
            window = idx[:, -bs:]
            logits, past, conf = self.forward_incremental(window, None, 0, streaming=True)
            pos = window.size(1)                            # absolute stream position of the next token
            tail = window[:, -m:] if m else None
            for step in range(max_new_tokens):
                next_tok = self._sample_next(
                    logits[:, -1, :], temperature, top_k, top_p, min_p,
                    conf[:, -1] if (use_confidence_temp and conf is not None) else None, conf_temp_min, conf_temp_max)
                if stop_on_eos:
                    next_tok = torch.where(finished.unsqueeze(-1), torch.full_like(next_tok, ByteTokenizer.EOS), next_tok)
                    finished = finished | (next_tok.squeeze(-1) == ByteTokenizer.EOS)
                yield next_tok, (conf[:, -1] if conf is not None else None)
                if (stop_on_eos and bool(finished.all())) or step == max_new_tokens - 1:
                    break
                logits, past, conf = self.forward_incremental(next_tok, past, pos, tail, streaming=True)
                pos += 1
                if m:
                    tail = torch.cat([tail, next_tok], dim=1)[:, -m:]
        finally:
            self.train(was_training)

    def _generate_iter(self, idx: torch.Tensor, max_new_tokens: int, temperature: float,
                       top_k: Optional[int], top_p: Optional[float], min_p: Optional[float],
                       use_confidence_temp: bool, conf_temp_min: float, conf_temp_max: float,
                       stop_on_eos: bool, use_cache: bool, rebuild_stride: int, cache_mode: str = "window"
                       ) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Yield ``(next_tok (B,1), confidence (B,) | None)`` for each generated position."""
        if cache_mode not in ("window", "streaming"):
            raise ValueError(f"cache_mode must be 'window' or 'streaming', got {cache_mode!r}")
        if cache_mode == "streaming" and not use_cache:
            raise ValueError("cache_mode='streaming' needs use_cache=True")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be None or positive, got {top_k}")
        if idx.dim() != 2 or idx.size(1) < 1:
            raise ValueError("idx must be (B, T) with T >= 1")
        bs = self.cfg.block_size
        if not 1 <= rebuild_stride <= bs:
            raise ValueError(f"rebuild_stride must be in [1, block_size={bs}], got {rebuild_stride}")
        K = self.cfg.byte_mix_kernel
        if cache_mode == "streaming":
            yield from self._generate_stream_iter(idx, max_new_tokens, temperature, top_k, top_p, min_p,
                                                  use_confidence_temp, conf_temp_min, conf_temp_max, stop_on_eos)
            return

        was_training = self.training
        self.eval()
        try:
            B = idx.size(0)
            finished = torch.zeros(B, dtype=torch.bool, device=idx.device) if stop_on_eos else None
            window = idx[:, -bs:]  # the tokens the next prediction is conditioned on
            past: Optional[List[KV]] = None
            for _ in range(max_new_tokens):
                if window.size(1) > bs:
                    # Context overflow: re-prefill from a trimmed window. stride=1 keeps exactly the
                    # trailing block_size tokens every step (== non-cached semantics, O(T^2)/step);
                    # a larger stride re-prefills only every `stride` tokens (amortised).
                    window = window[:, -(bs - rebuild_stride + 1):]
                    past = None
                n = window.size(1)
                if use_cache:
                    if past is None:
                        logits, past, conf = self.forward_incremental(window, None, 0)
                    else:
                        ctx = window[:, max(0, n - K): n - 1] if K and n > 1 else None
                        logits, past, conf = self.forward_incremental(window[:, -1:], past, n - 1, ctx)
                else:
                    logits, _, conf = self.forward(window)
                logits = logits[:, -1, :]
                conf = conf[:, -1] if conf is not None else None

                next_tok = self._sample_next(
                    logits, temperature, top_k, top_p, min_p,
                    conf if (use_confidence_temp and conf is not None) else None,
                    conf_temp_min, conf_temp_max)
                if stop_on_eos:
                    # freeze finished rows instead of letting them drift while others finish
                    next_tok = torch.where(finished.unsqueeze(-1),
                                           torch.full_like(next_tok, ByteTokenizer.EOS), next_tok)
                    finished = finished | (next_tok.squeeze(-1) == ByteTokenizer.EOS)
                window = torch.cat([window, next_tok], dim=1)
                yield next_tok, conf
                if stop_on_eos and bool(finished.all()):
                    break
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 min_p: Optional[float] = None, use_confidence_temp: bool = False,
                 conf_temp_min: float = 0.5, conf_temp_max: float = 1.5,
                 stop_on_eos: bool = True, use_cache: bool = True, rebuild_stride: int = 1,
                 return_confidence: bool = False, cache_mode: str = "window"):
        """Autoregressive sampling.

        Returns ``(B, T + new)`` ids (shorter if every row hit EOS; finished rows are padded
        with EOS). With ``return_confidence=True`` returns ``(ids, conf)`` where ``conf`` is
        (B, new) of confidence-head scores (None if the model has no head).

        use_cache: reuse a per-layer KV cache (bit-identical to ``use_cache=False`` -- see
            tests). ``rebuild_stride`` only matters once the running sequence exceeds
            ``block_size``: 1 reproduces the non-cached "last block_size tokens" semantics
            exactly; a larger value amortises the re-prefill (slightly shorter effective
            context right after each rebuild).
        cache_mode: what happens once prompt + output exceed ``block_size``.
            ``"window"`` (default) keeps the exact semantics of an uncached model fed the last
            ``block_size`` tokens: it re-prefills (O(block_size^2) work) as the window slides.
            ``"streaming"`` never re-prefills: it keeps a bounded per-layer KV cache (global layers: the
            last ``block_size`` keys; local layers: their window) at absolute RoPE positions, so each token
            costs O(1). It is NOT equivalent after overflow, for two reasons: (1) the cached states of retained
            tokens in layers above the first were computed while the tokens since dropped were still in view,
            whereas "window" recomputes them from the truncated text; (2) with ByteMix, "window" re-encodes the
            first tokens of its window from PAD left-context (as training windows start) while the cache holds
            their true left context -- visible to global layers. It is exactly equivalent while everything fits,
            and for a one-layer model whose attention does not reach the window start (any local layer; a global
            layer without ByteMix); otherwise measure the gap with :func:`tensegrity.evaluation.long_stream_nll`
            before relying on it.
        """
        toks, confs = [idx], []
        for next_tok, conf in self._generate_iter(
                idx, max_new_tokens, temperature, top_k, top_p, min_p, use_confidence_temp,
                conf_temp_min, conf_temp_max, stop_on_eos, use_cache, rebuild_stride, cache_mode):
            toks.append(next_tok)
            confs.append(conf)
        out = torch.cat(toks, dim=1)
        if not return_confidence:
            return out
        conf_out = torch.stack(confs, dim=1) if confs and confs[0] is not None else None
        return out, conf_out

    @torch.no_grad()
    def stream(self, idx: torch.Tensor, max_new_tokens: int, **kwargs) -> Iterator[Tuple[int, Optional[float]]]:
        """Yield ``(token_id, confidence)`` one byte at a time for a single sequence (B == 1).
        Accepts the sampling kwargs of :meth:`generate`."""
        if idx.size(0) != 1:
            raise ValueError("stream() supports a single sequence (batch size 1)")
        kw = dict(temperature=1.0, top_k=None, top_p=None, min_p=None, use_confidence_temp=False,
                  conf_temp_min=0.5, conf_temp_max=1.5, stop_on_eos=True, use_cache=True, rebuild_stride=1,
                  cache_mode="window")
        unknown = set(kwargs) - set(kw)
        if unknown:
            raise TypeError(f"stream() got unexpected keyword argument(s) {sorted(unknown)}")
        kw.update(kwargs)
        for next_tok, conf in self._generate_iter(idx, max_new_tokens, **kw):
            tid = int(next_tok[0, 0])
            yield tid, (float(conf[0]) if conf is not None else None)
            if kw["stop_on_eos"] and tid == ByteTokenizer.EOS:
                return

    # --------------------------------------------------------------- optimizer
    def configure_optimizer(self, weight_decay: float, learning_rate: float,
                            betas: Tuple[float, float], device_type: str, **kwargs) -> torch.optim.Optimizer:
        from .optim import configure_optimizer
        return configure_optimizer(self, weight_decay, learning_rate, betas, device_type, **kwargs)
