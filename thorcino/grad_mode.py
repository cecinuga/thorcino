"""Global switch controlling whether tensor operations record a backward node.

Kept in its own dependency-free module so `thorcino.tensor` can import it without
pulling in `thorcino.autograd` (which imports `thorcino.tensor` in turn)."""

_GRAD_ENABLED: bool = True

def is_grad_enabled() -> bool:
    """True while operations should wire their results into the autograd graph."""
    return _GRAD_ENABLED

class no_grad:
    """Context manager that stops operations from recording backward nodes.

    Inference and evaluation run under it so a forward pass allocates no graph:
    without it every eval batch builds - and immediately discards - a full
    backward graph, which for a recurrent model means one node per timestep."""

    def __enter__(self) -> "no_grad":
        global _GRAD_ENABLED
        self._previous: bool = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc_info: object) -> bool:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._previous
        return False
