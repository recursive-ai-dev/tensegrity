## 7 · Teach it something new without forgetting — measured, not assumed
Fine-tune on a *second* corpus and find out which way of doing it actually protects what the model knew. Three strategies start from the **same weights** and get the **same budget** (identical optimizer steps, tokens per step, schedule, seed and held-out windows):

* **new only** — every batch from the new corpus;
* **replay** — a fraction of every batch still comes from the original data;
* **replay + EWC** — replay plus Elastic Weight Consolidation (a quadratic penalty, weighted by an estimate of each weight's importance to the old data).

The table reports, per strategy, how much the *original* corpus got worse (**forgetting**) and how much the *new* one got better (**gain**) in bits per byte, and marks the strategies no other strategy beats on both. EWC is a loss term, not a guarantee: it only earns its place if it shows up here. One seed is an anecdote — list several in `CONTINUAL_SEEDS` to see the spread.
