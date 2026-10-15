## 6 · Climb the ladder — grow a trained model into the next size
Train a small model quickly, then **grow it into the next size and keep going**. Growth keeps the model's function *exactly* and gives every new head, hidden unit and layer a non-zero gradient path. Each growth step is a **transaction**: the candidate is built without touching the parent, every state entry of the parent is audited inside the grown model, and the candidate must preserve the function, survive save → reload (config + state dict, checkpoint, safetensors) and be able to learn *before* any compute is spent on it. Each rung is checkpointed and resumable; re-running this cell after a disconnect skips finished rungs.

**What to expect.** Growing preserves what was learned and lets you extend a run incrementally. In the repository's small CPU measurement (1M → 2M, equal compute) it did **not** train faster or reach a lower loss than training the 2M model from scratch, so use it for reuse and flexibility rather than as a shortcut; at larger scales that comparison is untested here.

Typical free-tier plan: `1M,2M,5M` with `0.15,0.3,1.0` hours.
