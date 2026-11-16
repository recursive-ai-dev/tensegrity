"""Byte-level tokenizer: 256 raw bytes + 3 specials + 6 ChatML control tokens. Zero training, zero dependencies."""

from __future__ import annotations

from typing import Any, List, Tuple


class ByteTokenizer:
    """
    Vocabulary = 256 raw byte values + 9 special tokens:
        256 = BOS (beginning of sequence)
        257 = EOS (end of sequence / document separator)
        258 = PAD (padding; also used as left-context filler by the byte-mix layer)
        259 = IM_START (ChatML turn start delimiter)
        260 = IM_END   (ChatML turn end delimiter)
        261 = ROLE_SYS (system role marker)
        262 = ROLE_USER (user role marker)
        263 = ROLE_ASST (assistant role marker)
        264 = ROLE_TOOL (tool role marker)

    Losslessness:
      * ``encode_bytes`` / ``decode_bytes`` are lossless for ANY bytes input.
      * ``encode`` / ``decode`` round-trip any well-formed ``str`` (UTF-8 is
        bijective on valid unicode). With the default ``errors="replace"`` they are
        NOT lossless for content strict UTF-8 cannot represent (e.g. lone
        surrogates); pass ``errors="strict"`` to fail loudly instead.

    ChatML Encoding:
      * Each turn encodes deterministically as: [IM_START, ROLE_*, ord('\n'), ...utf8_bytes..., IM_END, ord('\n')]
      * Conversation ends with EOS (257).
      * Loss mask (via ``encode_chat_with_mask``): 0 for system/user/tool turns and the assistant header
        [IM_START, ROLE_ASST, '\n']; 1 for assistant payload bytes and closing [IM_END, '\n'].
    """

    VOCAB_SIZE = 265
    BOS = 256
    EOS = 257
    PAD = 258
    IM_START = 259   # 
    IM_END = 260     # 
    ROLE_SYS = 261   # system
    ROLE_USER = 262  # user
    ROLE_ASST = 263  # assistant
    ROLE_TOOL = 264  # tool

    # Prepared data is ``bytes + EOS`` per document and contains no BOS, so the only start-of-document
    # context the model has ever seen as INPUT is EOS (the previous document's terminator). Start
    # generation from DOC_START: a BOS prompt is out-of-distribution (measured on a small model: ~3.1
    # bits/byte over a document's first 12 bytes after BOS vs ~0.3 after EOS).
    DOC_START = EOS
    _SPECIAL_IDS = frozenset((BOS, EOS, PAD, IM_START, IM_END, ROLE_SYS, ROLE_USER, ROLE_ASST, ROLE_TOOL))
    _SPECIAL_TOKEN_STRINGS = {BOS: "<|bos|>", EOS: "<|eos|>", PAD: "<|pad|>",
                               IM_START: "", IM_END: "",
                               ROLE_SYS: "<|system|>", ROLE_USER: "<|user|>", ROLE_ASST: "<|assistant|>", ROLE_TOOL: "<|tool|>"}

    # Role mapping for encode_chat
    _ROLE_TO_ID = {
        "system": ROLE_SYS,
        "user": ROLE_USER,
        "assistant": ROLE_ASST,
        "tool": ROLE_TOOL,
    }
    _ID_TO_ROLE = {v: k for k, v in _ROLE_TO_ID.items()}

    def __len__(self) -> int:
        return self.VOCAB_SIZE

    # ---------------------------------------------------------------- str <-> ids
    def encode(self, text: str, errors: str = "replace",
               add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8", errors=errors))
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    @staticmethod
    def _as_list(ids: Any) -> list:
        if hasattr(ids, "tolist"):
            return ids.tolist()
        return ids if isinstance(ids, (list, tuple)) else list(ids)

    def _validate(self, ids: list, who: str) -> None:
        bad = [int(i) for i in ids if not (0 <= int(i) < self.VOCAB_SIZE)]
        if bad:
            raise ValueError(f"{who}: invalid token id(s) outside vocab: {bad[:5]}"
                             f"{'...' if len(bad) > 5 else ''}")

    def decode(self, ids: Any, errors: str = "replace",
               skip_special_tokens: bool = True, strict: bool = False) -> str:
        """
        skip_special_tokens: drop ids >= 256 (default). If False, specials render as
            ``<|bos|>`` / ``<|eos|>`` / ``<|pad|>`` / ``<|im_start|>`` / ``<|im_end|>``
            / ``<|system|>`` / ``<|user|>`` / ``<|assistant|>`` / ``<|tool|>``.
        strict: raise ValueError on any id outside [0, VOCAB_SIZE) instead of dropping it.
        """
        ids = self._as_list(ids)
        if strict:
            self._validate(ids, "decode")
        if skip_special_tokens:
            return bytes(int(i) for i in ids if 0 <= int(i) <= 255).decode("utf-8", errors=errors)

        parts: List[str] = []
        cur = bytearray()
        for raw_id in ids:
            i = int(raw_id)
            if 0 <= i <= 255:
                cur.append(i)
            elif i in self._SPECIAL_TOKEN_STRINGS:
                if cur:
                    parts.append(cur.decode("utf-8", errors=errors))
                    cur.clear()
                parts.append(self._SPECIAL_TOKEN_STRINGS[i])
        if cur:
            parts.append(cur.decode("utf-8", errors=errors))
        return "".join(parts)

    # -------------------------------------------------------------- bytes <-> ids
    def encode_bytes(self, data: bytes, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"encode_bytes expects bytes/bytearray, got {type(data).__name__}")
        ids = list(data)
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode_bytes(self, ids: Any, strict: bool = False) -> bytes:
        """Strip specials, return raw bytes (no UTF-8 interpretation): the lossless path."""
        ids = self._as_list(ids)
        if strict:
            self._validate(ids, "decode_bytes")
        return bytes(int(i) for i in ids if 0 <= int(i) <= 255)

    # ---------------------------------------------------------------------- batch
    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    def decode_batch(self, batches: List[List[int]], **kwargs) -> List[str]:
        return [self.decode(ids, **kwargs) for ids in batches]

    # ------------------------------------------------------------- introspection
    def is_special(self, token_id: int) -> bool:
        return token_id in self._SPECIAL_IDS

    def special_tokens(self) -> Tuple[int, int, int]:
        return (self.BOS, self.EOS, self.PAD)

    # ---------------------------------------------------------------- ChatML helpers
    def encode_chat(self, messages: List[dict], add_eos: bool = True) -> List[int]:
        """Encode a list of ChatML messages into token IDs.

        Each message: {"role": "system"|"user"|"assistant"|"tool", "content": "..."}.
        Encoding per turn: [IM_START, ROLE_*, ord('\n'), ...utf8(content)..., IM_END, ord('\n')].
        Appends EOS at the end if add_eos=True.
        """
        out: List[int] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role not in self._ROLE_TO_ID:
                raise ValueError(f"Invalid role {role!r}; must be one of {list(self._ROLE_TO_ID)}")
            if not isinstance(content, str):
                raise ValueError(f"Message content must be str, got {type(content).__name__}")
            role_id = self._ROLE_TO_ID[role]
            out.extend([self.IM_START, role_id, ord('\n')])
            out.extend(content.encode("utf-8"))
            out.extend([self.IM_END, ord('\n')])
        if add_eos:
            out.append(self.EOS)
        return out

    def encode_chat_with_mask(self, messages: List[dict], add_eos: bool = True) -> Tuple[List[int], List[int]]:
        """Encode ChatML messages and produce a loss mask (same length).

        Mask = 0 for: system/user/tool turns entirely, and the assistant header [IM_START, ROLE_ASST, '\n'].
        Mask = 1 for: assistant payload bytes and the closing [IM_END, '\n'].
        EOS gets mask = 1 (model learns to emit EOS after assistant).
        """
        tokens: List[int] = []
        mask: List[int] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role not in self._ROLE_TO_ID:
                raise ValueError(f"Invalid role {role!r}; must be one of {list(self._ROLE_TO_ID)}")
            if not isinstance(content, str):
                raise ValueError(f"Message content must be str, got {type(content).__name__}")
            role_id = self._ROLE_TO_ID[role]
            # Header
            header = [self.IM_START, role_id, ord('\n')]
            tokens.extend(header)
            if role == "assistant":
                mask.extend([0, 0, 0])  # don't train on header
            else:
                mask.extend([0, 0, 0])  # don't train on system/user/tool at all
            # Content
            content_bytes = content.encode("utf-8")
            tokens.extend(content_bytes)
            if role == "assistant":
                mask.extend([1] * len(content_bytes))
            else:
                mask.extend([0] * len(content_bytes))
            # Footer
            footer = [self.IM_END, ord('\n')]
            tokens.extend(footer)
            if role == "assistant":
                mask.extend([1, 1])  # train on IM_END and newline
            else:
                mask.extend([0, 0])
        if add_eos:
            tokens.append(self.EOS)
            mask.append(1)  # train on EOS
        return tokens, mask

    @classmethod
    def upgrade_vocab(cls, new_vocab_size: int = 265) -> None:
        """Class method to upgrade VOCAB_SIZE for backward compatibility.

        This is used by TensegrityConfig.upgrade_vocab to expand the vocabulary
        while preserving legacy checkpoint compatibility.
        """
        if new_vocab_size < cls.VOCAB_SIZE:
            raise ValueError(f"Cannot downgrade vocab from {cls.VOCAB_SIZE} to {new_vocab_size}")
        cls.VOCAB_SIZE = new_vocab_size
        cls._SPECIAL_IDS = frozenset((cls.BOS, cls.EOS, cls.PAD,
                                       cls.IM_START, cls.IM_END,
                                       cls.ROLE_SYS, cls.ROLE_USER, cls.ROLE_ASST, cls.ROLE_TOOL))