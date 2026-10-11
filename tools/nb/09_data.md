## 2 · Data
Downloads are resumable and cached; preparing tokenises to a flat `uint16` file on local disk. `<|endoftext|>` separators in TinyStories become real **EOS tokens** (not 13 spelled-out bytes), which is what lets the model learn to end a story.
