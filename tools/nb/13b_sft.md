## 5B · Supervised Fine-Tuning (SFT): Pretrain → ChatML

If you already have a pretrained base checkpoint (from §4), you can continue training on a ChatML dataset. This **expands the vocabulary** (259 → 265 tokens for the ChatML control tokens), **widens the context** (1024 → 2048/4096), and **raises the RoPE base** (12k → 50k) so long conversations don't alias.

The SFT run:
1. Loads the base weights
2. Grows `token_emb` / `lm_head` to 265 tokens — new rows are initialized to the mean of existing embeddings + tiny orthogonal noise (so they train fast without logit spikes)
3. Updates `block_size`, `rope_base`, and sets `confidence_detach=True` for calibrated confidence
4. Verifies **forward parity** on standard byte inputs (0–255 + EOS) — the base model's behaviour on non-chat text is preserved
5. Trains with a lower LR, cosine decay, and `dropout=0.05` (SFT overfits narrow dialogue data easily)

**Use a ChatML dataset** (`format="chatml"` in §2) with `.mask` files so the loss is *assistant-only*. The cell below wires everything up; change `BASE_CKPT`, `SFT_HOURS`, and `SFT_LR` as needed.