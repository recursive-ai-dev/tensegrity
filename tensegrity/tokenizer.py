"""Byte-level tokenizer: 256 raw bytes + 3 specials. Zero training, zero dependencies."""

from __future__ import annotations

from typing import Any, List, Tuple


class ByteTokenizer:
    """
    Vocabulary = 256 raw byte values + 3 special tokens:
        256 = BOS (beginning of sequence)
        257 = EOS (end of sequence / document separator)
        258 = PAD (padding; also used as left-context filler by the byte-mix layer)

    Losslessness:
      * ``encode_bytes`` / ``decode_bytes`` are lossless for ANY bytes input.
      * ``encode`` / ``decode`` round-trip any well-formed ``str`` (UTF-8 is
        bijective on valid unicode). With the default ``errors="replace"`` they are
        NOT lossless for content strict UTF-8 cannot represent (e.g. lone
        surrogates); pass ``errors="strict"`` to fail loudly instead.
    """

    VOCAB_SIZE = 259
    BOS = 256
    EOS = 257
    PAD = 258
    _SPECIAL_IDS = frozenset((BOS, EOS, PAD))
    _SPECIAL_TOKEN_STRINGS = {BOS: "<|bos|>", EOS: "<|eos|>", PAD: "<|pad|>"}

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
            ``<|bos|>`` / ``<|eos|>`` / ``<|pad|>``.
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
