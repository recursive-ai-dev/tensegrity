## 7 · Interactive Multi-Turn Chat

The `ChatSession` class manages a stateful conversation with **persistent KV cache** across turns:
- **System prompt is pinned** in the KV cache (never evicted, even when context overflows)
- **Incremental turn processing** — only the new user message + assistant header are fed to the model; the rest stays cached
- **Sliding-window eviction** — when `max_history_tokens` is exceeded, the oldest complete user+assistant pair is evicted and the KV cache is rebuilt from the pinned system prompt + remaining recent turns
- **Streaming & batch generation** — `session.stream(user_text)` yields text as it's generated; `session.send(user_text)` returns the full response

### CLI equivalent
```bash
tensegrity chat --ckpt runs/5m/ckpt_best.pt \
                --system "You are a concise technical support assistant." \
                --temp 0.7 --min-p 0.05 --rep-penalty 1.12
```
Supports `/reset` (clears history, keeps system KV), `/stats` (prints cached tokens, KV memory, mean confidence), and `--json` for single-turn batch mode.