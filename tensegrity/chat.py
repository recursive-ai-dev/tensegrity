"""Stateful multi-turn ChatSession engine with persistent KV cache and system prompt pinning."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from .model import TensegrityLM
from .tokenizer import ByteTokenizer


class ChatSession:
    """Stateful multi-turn conversation manager with KV cache reuse.

    Features:
    - System prompt pinned in KV cache (never evicted)
    - Incremental turn processing (only new tokens fed to model)
    - Sliding window eviction of oldest turns when context overflows
    - Streaming and batch generation support
    """

    def __init__(self, model: TensegrityLM, system_prompt: str = "",
                 max_history_tokens: Optional[int] = None,
                 sampling_kwargs: Optional[Dict[str, Any]] = None):
        """
        Args:
            model: The TensegrityLM model (should be in eval mode)
            system_prompt: System prompt to prepend to every conversation
            max_history_tokens: Maximum tokens to keep in KV cache (default: model.block_size)
            sampling_kwargs: Default sampling parameters for generate/stream
        """
        self.model = model
        self.tokenizer = ByteTokenizer()
        self.system_prompt = system_prompt
        self.max_history_tokens = max_history_tokens or model.cfg.block_size
        self.sampling_kwargs = sampling_kwargs or {}

        # Conversation state
        self._past_kv: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
        self._pos = 0  # absolute stream position
        self._sys_len = 0  # length of system prompt in tokens
        self._turn_tokens: List[List[int]] = []  # tokens per turn (user + assistant)
        self._ctx_tokens: Optional[torch.Tensor] = None  # ByteMix context (K-1 tokens)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        self._system_prompt = value
        self.reset()

    def reset(self) -> None:
        """Clear conversation history but keep system prompt in KV cache."""
        self._past_kv = None
        self._pos = 0
        self._sys_len = 0
        self._turn_tokens = []
        self._ctx_tokens = None
        self._prefill_system()

    def _prefill_system(self) -> None:
        """Encode and prefill system prompt into KV cache."""
        if not self._system_prompt:
            self._sys_len = 0
            return

        # Encode system turn: [IM_START, ROLE_SYS, '\n', content, IM_END, '\n']
        sys_tokens = self.tokenizer.encode_chat([
            {"role": "system", "content": self._system_prompt}
        ], add_eos=False)  # Don't add EOS yet, we'll add it with the first user turn

        sys_ids = torch.tensor([sys_tokens], dtype=torch.long, device=next(self.model.parameters()).device)

        # Prefill system prompt
        self.model.eval()
        with torch.no_grad():
            logits, self._past_kv, _ = self.model.forward_incremental(sys_ids, None, 0)
        self._sys_len = len(sys_tokens)
        self._pos = self._sys_len

        # Initialize ByteMix context
        K = self.model.cfg.byte_mix_kernel
        if K > 1:
            self._ctx_tokens = sys_ids[:, -(K-1):] if sys_ids.size(1) >= K-1 else sys_ids
        else:
            self._ctx_tokens = None

    def _encode_user_turn(self, user_text: str) -> List[int]:
        """Encode user turn: [IM_START, ROLE_USER, '\n', content, IM_END, '\n', IM_START, ROLE_ASST, '\n']"""
        # User message
        user_tokens = self.tokenizer.encode_chat([
            {"role": "user", "content": user_text}
        ], add_eos=False)

        # Assistant header
        asst_header = [self.tokenizer.IM_START, self.tokenizer.ROLE_ASST, ord('\n')]

        return user_tokens + asst_header

    def _encode_assistant_closing(self) -> List[int]:
        """Encode assistant closing: [IM_END, '\n']"""
        return [self.tokenizer.IM_END, ord('\n')]

    def _evict_oldest_turn(self) -> None:
        """Evict the oldest complete user+assistant turn pair after system prompt."""
        if not self._turn_tokens:
            return

        # Find the first turn pair (user + assistant)
        # We need to re-prefill from system prompt + remaining turns
        evicted = self._turn_tokens.pop(0)
        self._rebuild_kv_cache()

    def _rebuild_kv_cache(self) -> None:
        """Rebuild KV cache from system prompt and remaining turns."""
        # Encode system + all remaining turns
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # Reconstruct conversation from turn tokens
        # This is a simplified approach - in practice we'd store the raw messages
        # For now, we'll re-prefill from the token stream
        all_tokens = []
        if self._system_prompt:
            sys_tokens = self.tokenizer.encode_chat([
                {"role": "system", "content": self._system_prompt}
            ], add_eos=False)
            all_tokens.extend(sys_tokens)

        for turn in self._turn_tokens:
            all_tokens.extend(turn)

        if not all_tokens:
            self._past_kv = None
            self._pos = 0
            self._ctx_tokens = None
            return

        # Find the cutoff point to fit in block_size
        max_tokens = self.max_history_tokens
        if len(all_tokens) > max_tokens:
            # Keep system prompt + most recent tokens
            sys_tokens = self.tokenizer.encode_chat([
                {"role": "system", "content": self._system_prompt}
            ], add_eos=False)
            sys_len = len(sys_tokens)
            # Keep system + recent tokens
            keep_tokens = all_tokens[:sys_len] + all_tokens[-(max_tokens - sys_len):]
            all_tokens = keep_tokens

        ids = torch.tensor([all_tokens], dtype=torch.long, device=next(self.model.parameters()).device)

        self.model.eval()
        with torch.no_grad():
            logits, self._past_kv, _ = self.model.forward_incremental(ids, None, 0)

        self._sys_len = len(sys_tokens) if self._system_prompt else 0
        self._pos = len(all_tokens)

        # Update ByteMix context
        K = self.model.cfg.byte_mix_kernel
        if K > 1:
            self._ctx_tokens = ids[:, -(K-1):] if ids.size(1) >= K-1 else ids
        else:
            self._ctx_tokens = None

    def _maybe_evict(self) -> None:
        """Check if we need to evict old turns due to context overflow."""
        if self._pos >= self.max_history_tokens:
            self._evict_oldest_turn()

    def send(self, user_text: str, **sampling_kwargs) -> str:
        """Send a user message and get the assistant's response.

        Args:
            user_text: User's message
            **sampling_kwargs: Override default sampling parameters

        Returns:
            Assistant's response text
        """
        # Combine default and override sampling kwargs
        kwargs = {**self.sampling_kwargs, **sampling_kwargs}

        # Encode user turn + assistant header
        turn_tokens = self._encode_user_turn(user_text)
        turn_ids = torch.tensor([turn_tokens], dtype=torch.long, device=next(self.model.parameters()).device)

        # Feed new tokens incrementally
        self.model.eval()
        with torch.no_grad():
            logits, self._past_kv, _ = self.model.forward_incremental(
                turn_ids, self._past_kv, self._pos, self._ctx_tokens
            )
        self._pos += len(turn_tokens)

        # Generate assistant response
        generated_tokens = []
        for next_tok, _ in self.model._generate_iter(
            turn_ids[:, -1:],  # Start from last token of assistant header
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_k=kwargs.get("top_k", 40),
            top_p=kwargs.get("top_p", None),
            min_p=kwargs.get("min_p", 0.05),
            use_confidence_temp=kwargs.get("use_confidence_temp", False),
            conf_temp_min=kwargs.get("conf_temp_min", 0.5),
            conf_temp_max=kwargs.get("conf_temp_max", 1.5),
            stop_on_eos=True,
            use_cache=True,
            rebuild_stride=1,
            cache_mode="window",
            repetition_penalty=kwargs.get("repetition_penalty", 1.12),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            penalty_window=kwargs.get("penalty_window", 256),
            stop_token_ids=(self.tokenizer.EOS, self.tokenizer.IM_END),
            stop_strings=kwargs.get("stop_strings", ())
        ):
            generated_tokens.append(next_tok.item())
            # Check for stop tokens
            if next_tok.item() in (self.tokenizer.EOS, self.tokenizer.IM_END):
                break

        # Append assistant closing
        closing = self._encode_assistant_closing()
        closing_ids = torch.tensor([closing], dtype=torch.long, device=next(self.model.parameters()).device)
        self.model.eval()
        with torch.no_grad():
            logits, self._past_kv, _ = self.model.forward_incremental(
                closing_ids, self._past_kv, self._pos, self._ctx_tokens
            )
        self._pos += len(closing)

        # Store turn tokens for potential eviction
        full_turn = turn_tokens + generated_tokens + closing
        self._turn_tokens.append(full_turn)

        # Update ByteMix context
        K = self.model.cfg.byte_mix_kernel
        if K > 1:
            all_recent = turn_ids[:, -1:] if turn_ids.size(1) > 0 else torch.empty(1, 0, dtype=torch.long, device=turn_ids.device)
            # Simplified: just track the last K-1 tokens
            recent = torch.cat([turn_ids, torch.tensor([generated_tokens + closing], device=turn_ids.device)], dim=1)
            self._ctx_tokens = recent[:, -(K-1):] if recent.size(1) >= K-1 else recent

        # Check for eviction
        self._maybe_evict()

        # Decode assistant response (skip the header we added)
        response_tokens = generated_tokens
        # Find IM_END or EOS
        end_idx = len(response_tokens)
        for i, t in enumerate(response_tokens):
            if t in (self.tokenizer.EOS, self.tokenizer.IM_END):
                end_idx = i
                break
        response_tokens = response_tokens[:end_idx]
        return self.tokenizer.decode(response_tokens)

    def stream(self, user_text: str, **sampling_kwargs) -> Iterator[str]:
        """Stream assistant response token by token.

        Yields:
            Decoded text chunks as they are generated
        """
        kwargs = {**self.sampling_kwargs, **sampling_kwargs}

        # Encode user turn + assistant header
        turn_tokens = self._encode_user_turn(user_text)
        turn_ids = torch.tensor([turn_tokens], dtype=torch.long, device=next(self.model.parameters()).device)

        # Feed new tokens incrementally
        self.model.eval()
        with torch.no_grad():
            logits, self._past_kv, _ = self.model.forward_incremental(
                turn_ids, self._past_kv, self._pos, self._ctx_tokens
            )
        self._pos += len(turn_tokens)

        # Stream assistant response
        decoder = torch.text.codecs.getdecoder("utf-8")() if hasattr(torch.text, "codecs") else None
        # Simple byte-by-byte decoding
        generated_tokens = []
        for next_tok, _ in self.model._generate_iter(
            turn_ids[:, -1:],
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_k=kwargs.get("top_k", 40),
            top_p=kwargs.get("top_p", None),
            min_p=kwargs.get("min_p", 0.05),
            use_confidence_temp=kwargs.get("use_confidence_temp", False),
            conf_temp_min=kwargs.get("conf_temp_min", 0.5),
            conf_temp_max=kwargs.get("conf_temp_max", 1.5),
            stop_on_eos=True,
            use_cache=True,
            rebuild_stride=1,
            cache_mode="window",
            repetition_penalty=kwargs.get("repetition_penalty", 1.12),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            penalty_window=kwargs.get("penalty_window", 256),
            stop_token_ids=(self.tokenizer.EOS, self.tokenizer.IM_END),
            stop_strings=kwargs.get("stop_strings", ())
        ):
            tid = next_tok.item()
            generated_tokens.append(tid)

            # Decode incrementally
            if tid < 256:
                yield chr(tid)

            # Check for stop tokens
            if tid in (self.tokenizer.EOS, self.tokenizer.IM_END):
                break

        # Append assistant closing
        closing = self._encode_assistant_closing()
        closing_ids = torch.tensor([closing], dtype=torch.long, device=next(self.model.parameters()).device)
        self.model.eval()
        with torch.no_grad():
            logits, self._past_kv, _ = self.model.forward_incremental(
                closing_ids, self._past_kv, self._pos, self._ctx_tokens
            )
        self._pos += len(closing)

        # Store turn tokens
        full_turn = turn_tokens + generated_tokens + closing
        self._turn_tokens.append(full_turn)

        # Update ByteMix context
        K = self.model.cfg.byte_mix_kernel
        if K > 1:
            recent = torch.tensor([generated_tokens + closing], device=turn_ids.device)
            self._ctx_tokens = recent[:, -(K-1):] if recent.size(1) >= K-1 else recent

        self._maybe_evict()

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        total_cached = self._pos if self._past_kv is not None else 0
        kv_memory = 0
        if self._past_kv:
            for kv in self._past_kv:
                if kv is not None:
                    k, v = kv
                    kv_memory += k.numel() * k.element_size() + v.numel() * v.element_size()

        return {
            "cached_tokens": total_cached,
            "system_tokens": self._sys_len,
            "turn_count": len(self._turn_tokens),
            "kv_memory_mb": kv_memory / 1e6,
            "max_history_tokens": self.max_history_tokens,
        }


def create_chat_session(ckpt_path: str, system_prompt: str = "",
                        device: str = "auto", **sampling_kwargs) -> ChatSession:
    """Convenience function to create a ChatSession from a checkpoint."""
    from .checkpoint import load_model
    from .hw import pick_device

    dev = pick_device(device)
    model, cfg, _ = load_model(ckpt_path, dev)
    model.eval()
    return ChatSession(model, system_prompt, sampling_kwargs=sampling_kwargs)