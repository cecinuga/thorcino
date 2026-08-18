[loader](core/loader.py)
1) add parallel loading via multi threading
2) add pre fetching of N+1 batches

[autograd](core/autograd/arithmetic.py)
[autograd](core/autograd/activations.py)
[autograd](core/autograd/losses.py)
1) Makes backward pass computation over numpy arrays and let Tensor class be only the frontend
2) Add Cache: instend of recomputing base functions, cache it from forward pass and let backward pass reuse it
3) Add Debug step when backward fails: show on which node the failure occurred

Test