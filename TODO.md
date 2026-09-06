[trainer](thorcino/training/trainer.py)
1) create a statich method in order to create a new Trainer instance directly from a checkpoint file.

[loader](thorcino/loader.py)
1) add parallel loading via multi threading
2) add pre fetching of N+1 batches

[autograd](thorcino/autograd/arithmetic.py)
[autograd](thorcino/autograd/activations.py)
[autograd](thorcino/autograd/losses.py)
2) Add Cache: instend of recomputing base functions, cache it from forward pass and let backward pass reuse it
3) Add Debug step when backward fails: show on which node the failure occurred