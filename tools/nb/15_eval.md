## 5 · Evaluate & talk to it
**Bits per byte** is the tokenizer-independent number: the model's total negative log-likelihood of the held-out *bytes* divided by their count (document-boundary EOS tokens are accounted for separately, so they don't flatter or penalise it). Perplexity here is per *byte token* — don't compare it with a subword model's.

The **context probe** measures what the model actually *uses* of its context: it plants a random key, buries it under `d` unrelated letters, repeats its start, and checks whether the model can finish it. Read it next to the attention reach printed above it — a layout that cannot reach a distance cannot beat chance there; a model that *can* reach it still has to have learned to.
