## 4 · Train
Run this cell as often as you like — it resumes automatically and **exactly**: weights, optimizer, EMA, every RNG and the data order are restored, and the cell refuses (with a message that says what differs) if the model, the training settings or the data are not the ones the run started with. Changing settings on purpose? Start a new run from the weights (`INIT_FROM_WEIGHTS`) so the change is explicit.

The dashboard shows **bits per byte** (lower is better; 8.0 is chance, ≈1.0–1.5 is good story text), the learning-rate schedule and throughput. Every evaluation also prints a short sample so you can *read* the progress.
