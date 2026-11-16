Architectural Blueprint: Refactoring Tensegrity into a Byte-Level Conversational Engine
1. Architectural Shift Overview
Currently, Tensegrity operates as a unidirectional document-continuation model:
Vocabulary (vocab_size = 259): Raw UTF-8 bytes 0..255 plus BOS (256), EOS (257), and PAD (258).
Data Stream: Flat uint16 arrays where documents are concatenated as <bytes> + EOS, and ByteDataset samples random contiguous block_size windows across document boundaries without masking.
Loss Function: Uniform next-byte cross-entropy across every position in targets.
Inference (generate / stream): Stateless prompt completion starting from DOC_START (EOS) without turn boundaries, role tracking, or multi-turn KV-cache reuse.
Transforming Tensegrity into a ChatML-native conversational engine requires six coordinated changes across the package while preserving backward compatibility for base-model pretraining checkpoints and the grow_model ladder:
Layer
Current (Completion LM)
Target (Conversational Engine)
Tokenizer (tokenizer.py)
256 bytes + BOS, EOS, PAD (259)
+ Atomic ChatML control tokens (IM_START, IM_END, SYS, USER, ASST, TOOL) (265)
Ingestion (ingest.py)
Raw text / single-field JSONL  train.bin
Multi-turn messages JSONL  train.bin + parallel train.mask (assistant-only loss mask)
Batching (data.py)
Unmasked random slices across EOS
Intra-document causal attention masks + loss-mask alignment (targets = -1 on prompt turns)
Architecture (config.py, layers.py, model.py)
block_size = 1024, rope_base = 12000, raw argmax confidence
block_size = 2048/4096, rope_base = 50000, intra-doc SDPA masking, detached soft-calibrated confidence
Sampling (model.py)
Temp, Top-K, Top-P, Min-P
+ Presence/frequency/repetition penalties + multi-token stop sequences (`<
Runtime (chat.py, cli.py, colab.py)
Single-shot generate
Stateful ChatSession with persistent KV cache, turn eviction/sliding, and interactive CLI/UI

2. File-by-File Implementation Specification
2.1 tensegrity/tokenizer.py — Atomic ChatML Control Tokens
Spelling <|im_start|>assistant\n in raw UTF-8 wastes 22 bytes per turn (~88 bytes across a 2-turn exchange—nearly 9% of a 1024-byte context window) and forces small models (1M–10M) to spend capacity predicting markup syntax.
Vocabulary Expansion (VOCAB_SIZE = 265):
Python
VOCAB_SIZE = 265
BOS = 256
EOS = 257
PAD = 258
IM_START = 259   # <|im_start|>
IM_END = 260     # <|im_end|>
ROLE_SYS = 261   # system
ROLE_USER = 262  # user
ROLE_ASST = 263  # assistant
ROLE_TOOL = 264  # tool


ChatML Turn Serialization (encode_chat): Each turn encodes deterministically as:[IM_START, ROLE_*, ord('\n'), ...utf8_bytes..., IM_END, ord('\n')] Followed by EOS (257) at the end of the conversation.
Loss Mask Generation (encode_chat_with_mask): Simultaneously emit a uint8 mask array of identical length:
0 (ignore in loss) for system, user, tool turns, and the [IM_START, ROLE_ASST, ord('\n')] header.
1 (compute loss) for the assistant's payload bytes and the closing [IM_END, ord('\n')] (so the model learns when to stop speaking).
Backward Compatibility: Keep VOCAB_SIZE = 259 as the default in TensegrityConfig for legacy checkpoints, and add TensegrityConfig.upgrade_vocab(new_vocab_size=265) (using grow_model logic: zero-initializing new embedding rows or initializing them to the mean of existing embeddings so pretrained base models adapt smoothly during SFT).
2.2 tensegrity/ingest.py — Multi-Turn ChatML Ingestion & Loss Masks
Extend prepare_data to ingest standard conversational JSONL ({"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}) alongside raw text corpora.
New Ingestion Mode (format="chatml"):
Validate each record has a non-empty messages list of dicts with valid role and non-empty string content.
Enforce structural rules (e.g., at least one assistant turn; reject malformed turns into _Rejects with explicit reason codes like no_assistant_turn, invalid_role).
Encode tokens into train.bin / val.bin (uint16) and write a parallel bit-packed or uint8 loss mask file (train.mask / val.mask) of exact equal token length.
Record "has_loss_mask": True and "chatml_schema": "v1" in meta.json and include train.mask / val.mask SHA-256 digests inside compute_fingerprint (manifest.py).
Mixed Pretrain + SFT Support: When ingesting raw prose (format="text"), omit .mask (or treat all non-PAD tokens as 1), allowing ByteDataset to seamlessly mix raw prose pretraining directories and ChatML SFT directories in the same Trainer run.
2.3 tensegrity/data.py — Assistant-Only Targets & Intra-Document Isolation
In conversational training, two bugs degrade small models rapidly: training on user prompts (wasting capacity memorizing user phrasing) and cross-conversation attention bleed.
Loss Mask Application in _gather: If mask.bin exists for a dataset directory, load it as np.memmap(..., dtype=np.uint8) alongside train.bin. In _gather:
Python
xy = arr[idx].astype(np.int64)
x, y = xy[:, :-1], xy[:, 1:]
if mask_arr is not None:
    m = mask_arr[idx[:, 1:]]  # target positions
    y = np.where(m == 1, y, -1)

Because model.py already passes ignore_index=-1 to F.cross_entropy, zero changes are needed in the loss computation itself—user and system tokens automatically produce zero gradient while still conditioning the KV states for assistant turns.
Intra-Document Causal Mask (doc_ids): Compute document boundaries on the fly inside _gather:
Python
# Increment doc_id right AFTER every EOS token so EOS belongs to the ending document
is_eos = (x == ByteTokenizer.EOS)
doc_ids = np.cumsum(is_eos, axis=1)
doc_ids = np.concatenate([np.zeros((x.shape[0], 1), dtype=np.int64), doc_ids[:, :-1]], axis=1)

Return (x, y, doc_ids) from get_batch and fixed_batches.
2.4 tensegrity/config.py, layers.py & model.py — Long-Context & Attention Isolation
To support multi-turn dialogue without losing early system instructions:
Context & RoPE Scaling (config.py, presets.py):
Add conversational preset variants (5m-chat, 10m-chat, 50m-chat, 115m-chat) with:
block_size = 2048 (for 5M/10M) and 4096 (for 50M/115M) — giving ~450 to ~900 words of active context.
rope_base = 50000.0 (preventing high-frequency RoPE aliasing across multi-kilobyte conversations).
vocab_size = 265.
Intra-Document Attention Masking (layers.py):
In CausalSelfAttention.forward, accept optional doc_ids: Optional[torch.Tensor] = None of shape (B, T).
When doc_ids is not None during training:
Global layers: Construct a causal intra-document boolean mask attn_mask = (doc_ids[:, None, :, None] == doc_ids[:, None, None, :]) & causal_tril ((B, 1, T, T)) and pass it to F.scaled_dot_product_attention.
Local banded layers: Slice doc_ids into [prev_chunk | curr_chunk] of shape (B * n_chunks, 1, 1, 2W) and AND it with _band_mask(n_chunks, W, B, q.device).
Result: Even when 4 short customer-service dialogues are packed into a single 2048-byte training window, zero attention weight ever crosses an EOS boundary.
Fused ByteMix Contraction (layers.py):
Replace the Python loop for j in range(1, self.kernel) in ByteMix.forward with e.unfold(dimension=1, size=self.kernel, step=1) so all K-1 taps are contracted in one vectorized tensor operation (torch.einsum), cutting small-model CPU launch overhead and avoiding ByteMix bleed across EOS when doc_ids changes.
Calibrated Confidence Head (model.py):
Set confidence_detach = True by default on chat presets so auxiliary confidence gradients never distort dialogue representations.
Replace binary (logits.argmax(-1) == targets) supervision with soft target probability:
Python
with torch.no_grad():
    probs = torch.softmax(logits.float(), dim=-1)
    target_prob = probs.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)

This gives smooth, well-calibrated uncertainty scores during multi-turn sampling.
2.5 tensegrity/model.py — Repetition Penalties & Multi-Token Stop Sequences
Byte-level models are uniquely prone to character- and phrase-level loops ("I would be happy to help you with that. I would be happy to help you...") when temperature is lowered for factual or customer-service consistency.
Sampling Penalties in _sample_next: Add repetition_penalty: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, and penalty_window: int = 256:
Track token counts over the trailing penalty_window tokens of window.
Apply HuggingFace-style multiplicative repetition_penalty (dividing positive logits and multiplying negative logits for seen tokens) plus OpenAI-style additive frequency_penalty * count + presence_penalty * (count > 0).
Structured Stop Criteria (stop_tokens / stop_strings):
Add stop_token_ids: Tuple[int, ...] = (ByteTokenizer.EOS, ByteTokenizer.IM_END) to generate() and stream().
Support multi-byte string stop sequences (e.g., "\nUser:", "\n<|im_end|>") via a rolling tail buffer in _generate_iter so generation halts cleanly at turn boundaries without leaking the next turn's role header.
2.6 tensegrity/chat.py — Stateful Multi-Turn ChatSession Engine
Create a dedicated runtime module tensegrity/chat.py that manages conversation state, system prompt pinning, and incremental KV-cache reuse across turns.



Python
class ChatSession:
    def __init__(self, model: TensegrityLM, system_prompt: str = "",
                 max_history_tokens: Optional[int] = None,
                 sampling_kwargs: Optional[Dict[str, Any]] = None): ...


Zero-Redundant-Prefill Turn Management:
On initialization, encode [IM_START, ROLE_SYS, '\n', ...system_bytes..., IM_END, '\n'] and prefill the KV cache once, recording sys_len.
On session.send(user_text):
Encode [IM_START, ROLE_USER, '\n', ...user_bytes..., IM_END, '\n', IM_START, ROLE_ASST, '\n'].
Incrementally feed only those new user/header tokens into model.forward_incremental (reusing the cached KV states from all prior turns!).
Stream generated assistant tokens until IM_END or EOS is emitted.
Append [IM_END, '\n'] to the KV cache so the assistant's closing delimiter is already cached for the next user turn.
System-Prompt-Pinned Context Eviction:
When total_cached_tokens > block_size:
Never drop the first sys_len tokens (the system persona / customer-service policy rules stay pinned at positions 0..sys_len-1).
Evict the oldest complete user+assistant turn pair after sys_len, re-prefill the pinned system prompt + remaining recent turns cleanly, and continue—eliminating the stale-KV drift of raw streaming mode while keeping the system prompt intact forever.
2.7 tensegrity/train.py & workflow.py — Two-Stage Pretrain  SFT Pipeline
Add a first-class SFT workflow helper in tensegrity/workflow.py:
prepare_sft_from_base(base_ckpt_path, new_vocab_size=265, new_block_size=2048):
Loads a pretrained base checkpoint (e.g., 10m or 50m), expands token_emb / lm_head to accommodate the ChatML control tokens (initializing new control token embeddings to the normalized mean of existing embeddings + small orthogonal noise so they train rapidly without logit spikes), updates block_size and rope_base, and verifies forward parity on standard byte inputs.
SFT TrainConfig Defaults:
Lower peak learning rate (learning_rate = 1e-4 for AdamW, muon_lr = 0.005 for Muon), cosine decay schedule (schedule = "cosine"), dropout = 0.05 (to prevent overfitting narrow dialogue datasets), and eval_suite tracking both retain (base prose bpb_bytes) and adapt (ChatML assistant-turn bpb_bytes).
2.8 tensegrity/cli.py & notebooks/Tensegrity_Colab.ipynb — Interactive Chat & Deployment
CLI (tensegrity chat):
Bash
tensegrity chat --ckpt runs/10m_chat/ckpt_best.pt \
                --system "You are a concise, helpful technical support assistant." \
                --temp 0.7 --min-p 0.05 --rep-penalty 1.12

Supports interactive REPL streaming, /reset (clears history while keeping system KV cache), /stats (prints cached tokens, KV memory, and mean token confidence), and --json single-turn batch mode for local API/daemon integration.
Colab Notebook (§1–§7 Additions):
§2 Data: Toggle between Raw Corpus (Pretrain) and ChatML Conversations (SFT JSONL) with an automatic ChatML audit view showing [MASKED] user turns vs [TRAINED] assistant turns.
§5B SFT Fine-Tune Cell: Optional one-click transition from the base pretrained checkpoint to ChatML SFT.
§7 Interactive Chat Cell: Replaces the static prompt box with a multi-turn chat loop using ChatSession, confidence-colored token rendering, and one-click export (safetensors + tokenizer_config.json with ChatML template metadata).
3. Recommended Execution Order
Step 1 (Data & Tokenizer): Implement IM_START..ROLE_TOOL in tokenizer.py, .mask emission in ingest.py, and doc_ids + y = -1 masking in data.py.
Step 2 (Model & Attention): Add doc_ids intra-document masking in layers.py, soft confidence calibration + repetition penalties in model.py, and *-chat presets (block_size=2048, rope_base=50000) in presets.py.
Step 3 (Session Runtime & CLI): Build tensegrity/chat.py (ChatSession with pinned system prompt eviction) and wire tensegrity chat into cli.py.
Step 4 (Verification Suite): Add tests/test_chat.py verifying:
Zero loss gradient on system/user turns when mask is active.
Zero cross-document attention leakage across EOS boundaries.
Bit-exact parity between incremental multi-turn ChatSession KV decoding and full-conversation forward passes.
System-prompt retention after context overflow eviction.
