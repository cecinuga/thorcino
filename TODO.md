[scaling_law](./examples/recurrent/scaling_law/main.ipynb)
add the recover of backups and metrics support

[loader](core/loader.py)
1) add parallel loading via multi threading
2) add pre fetching of N+1 batches

[autograd](core/autograd/arithmetic.py)
[autograd](core/autograd/activations.py)
[autograd](core/autograd/losses.py)
2) Add Cache: instend of recomputing base functions, cache it from forward pass and let backward pass reuse it
3) Add Debug step when backward fails: show on which node the failure occurred