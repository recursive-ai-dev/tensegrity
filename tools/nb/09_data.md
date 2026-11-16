## 2 · Data
Downloads are resumable, cached and hash-recorded; preparing turns **documents** into tokens. The corpus is split into train and validation *by document* (a hash of each document decides, so the same document always lands on the same side and exact duplicates can never straddle the split); a corpus with too few documents to split that way (e.g. one big text file) falls back to a contiguous tail split, and the cell says so. `<|endoftext|>` separators in TinyStories become real **EOS tokens** (not 13 spelled-out bytes), which is what lets the model learn to end a story.

### ChatML format (for supervised fine-tuning)
For conversational training, use `format="chatml"` with a JSONL file where each line is:
```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
```
This produces `train.bin` / `val.bin` **plus** parallel `train.mask` / `val.mask` (uint8) files. The mask marks assistant payload bytes + closing `IM_END\n` with `1` (compute loss) and everything else with `0` (ignore in loss). The model still conditions on the full conversation — it just doesn't spend capacity predicting the user's words or the markup. The manifest records `has_loss_mask: true` and `chatml_schema: "v1"`.

The prepared directory carries a **manifest**: SHA-256 of both token files, the split policy, counts, and a *fingerprint* that names exactly which bytes the model will see. Training records that fingerprint, so resuming on different data is refused instead of silently allowed. The audit below shows the fingerprint, integrity and decoded examples from both splits — **read them** before spending hours.