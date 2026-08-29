from __future__ import annotations
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import graphviz

from thorcino.tensor import Tensor

if TYPE_CHECKING:
    from thorcino.layers import Layer
    from thorcino.autograd import Function


# Visual palette shared by every node kind so the two sub-graphs read as one picture.
_STYLE = {
    "layer":  {"shape": "record", "style": "filled", "fillcolor": "#cfe8ff", "color": "#2b6cb0"},
    "io":     {"shape": "ellipse", "style": "filled", "fillcolor": "#e2e8f0", "color": "#4a5568"},
    "op":     {"shape": "box", "style": "rounded,filled", "fillcolor": "#fed7aa", "color": "#c05621"},
}

# Each tensor node is labelled and coloured by its semantic role. The four
# roles are: input, weights, bias (leaf parameters/inputs) and hidden (any
# tensor produced by an operation).
_ROLE_STYLE = {
    "input":   {"shape": "ellipse", "style": "filled", "fillcolor": "#e2e8f0", "color": "#4a5568"},
    "weights": {"shape": "ellipse", "style": "filled", "fillcolor": "#c6f6d5", "color": "#2f855a"},
    "bias":    {"shape": "ellipse", "style": "filled", "fillcolor": "#bee3f8", "color": "#2b6cb0"},
    "hidden":  {"shape": "ellipse", "style": "filled", "fillcolor": "#faf089", "color": "#975a16"},
}


class ComputationalGraph:
    """Builds a graphviz picture of a Sequential model.

    The picture can hold up to three clusters:
      * "Network Architecture" - one node per layer with its basic info,
        annotated with the real input/output shape it produces.
      * "Forward Computational Graph" - the operation graph produced by a
        forward pass with a real input shape, showing how data flows from
        the input to the output.
      * "Backward Computational Graph" - the same autograd graph read in
        reverse, showing how gradients flow between tensors.

    Every cluster is built from a single real forward/backward pass run with
    the ``shape`` given to :meth:`build`, so the tensors on the picture carry
    the exact shapes the model would see in production - not a synthetic
    stand-in.
    """

    def __init__(self, model: "Layer"):
        self.model = model
        self._graph: graphviz.Digraph | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self, shape: tuple[int, ...], arch: bool = True, forward: bool = False,
              backward: bool = False) -> None:
        """Create the graphviz graph and cache it on the instance.

        Args:
            shape: real input shape (including the batch dimension) to feed
                the model with, e.g. ``(batch, in_feature)`` for a plain
                feed-forward model or ``(batch, seq_len, in_feature)`` for a
                recurrent one. Drives a real forward/backward pass so every
                node on the picture is annotated with its actual shape.
            arch: include the "Network Architecture" cluster (layer info).
            forward: include the "Forward Computational Graph" cluster
                (data flow from input to output).
            backward: include the "Backward Computational Graph" cluster
                (autograd operations produced by a forward/backward pass).
        """
        if not arch and not forward and not backward:
            raise ValueError("build() needs at least one of arch/forward/backward to be True")

        graph = graphviz.Digraph("model", format="png")
        graph.attr(rankdir="TB", fontname="Helvetica", labelloc="t",
                   label="tiny-torch model graph", fontsize="18")
        graph.attr("node", fontname="Helvetica", fontsize="11")
        graph.attr("edge", fontname="Helvetica", fontsize="9", color="#4a5568")

        # A single real pass feeds every cluster so the arch/forward/backward
        # pictures all agree on the same shapes and the same tensors.
        out, layer_io = self._run_forward(shape)

        if arch:
            self._build_network(graph, shape, layer_io)
        if forward and out is not None:
            self._build_forward(graph, out)
        if backward and out is not None:
            self._build_backward(graph, out)

        self._graph = graph

    def render(self, path: str | Path) -> None:
        """Render the previously built graph to a .png at ``path``."""
        if self._graph is None:
            raise RuntimeError("graph not built yet: call build_graph() first")

        path = Path(path)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)

        # pipe() returns the rendered bytes so we control the exact output path
        # and avoid graphviz's intermediate .gv source file.
        path.write_bytes(self._graph.pipe(format="png"))

    # ------------------------------------------------------------------ #
    # Network architecture cluster
    # ------------------------------------------------------------------ #
    def _build_network(self, graph: graphviz.Digraph, shape: tuple[int, ...],
                        layer_io: list[tuple["Layer", tuple[int, ...], tuple[int, ...]]]) -> None:
        with graph.subgraph(name="cluster_network") as net:
            net.attr(label="Network Architecture", style="rounded",
                     color="#2b6cb0", fontcolor="#2b6cb0", fontsize="14")

            prev = "net_input"
            net.node(prev, f"input\n{tuple(shape)}", **_STYLE["io"])

            out_shape = tuple(shape)
            for i, (layer, in_shape, layer_out_shape) in enumerate(layer_io):
                node_id = f"layer_{i}"
                net.node(node_id, self._layer_label(layer, in_shape, layer_out_shape), **_STYLE["layer"])
                net.edge(prev, node_id)
                prev = node_id
                out_shape = layer_out_shape

            net.node("net_output", f"output\n{out_shape}", **_STYLE["io"])
            net.edge(prev, "net_output")

    _REPR_RE = re.compile(r"^(\w+)\((.*)\)$")

    @classmethod
    def _layer_label(cls, layer: "Layer", in_shape: tuple[int, ...] | None = None,
                      out_shape: tuple[int, ...] | None = None) -> str:
        """Record-shaped label built from the layer's own ``__repr__``,
        annotated with the real shape it consumed/produced during the run."""
        text = repr(layer)
        match = cls._REPR_RE.match(text)
        if not match:
            fields = [text]
        else:
            name, body = match.groups()
            fields = [name] + cls._split_top_level(body) if body else [name]

        if in_shape is not None:
            fields.append(f"in: {tuple(in_shape)}")
        if out_shape is not None:
            fields.append(f"out: {tuple(out_shape)}")

        # graphviz record: fields separated by '|'
        return "{" + " | ".join(fields) + "}"

    @staticmethod
    def _split_top_level(body: str) -> list[str]:
        """Split a repr's argument body on top-level commas, ignoring
        commas nested inside parens/brackets/braces (e.g. a Tensor repr)."""
        fields: list[str] = []
        depth = 0
        start = 0
        for i, ch in enumerate(body):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                fields.append(body[start:i].strip())
                start = i + 1
        fields.append(body[start:].strip())
        return fields

    # ------------------------------------------------------------------ #
    # Forward computational graph cluster
    # ------------------------------------------------------------------ #
    def _build_forward(self, graph: graphviz.Digraph, out: Tensor) -> None:
        if out._grad_fn is None:
            return

        with graph.subgraph(name="cluster_forward") as fwd:  # pyright: ignore[reportUnknownMemberType, reportOptionalContextManager]
            fwd.attr(label="Forward Computational Graph", style="rounded",
                     color="#2f855a", fontcolor="#2f855a", fontsize="14")

            self._tensor_node(fwd, out, "f")
            for t in self._topo_order(out):
                fn = t._grad_fn
                assert fn is not None  # _topo_order only yields produced tensors
                fn_id = f"ff{id(fn)}"
                fwd.node(fn_id, self._op_forward_name(fn), **_STYLE["op"])
                # The operation produces its result -> data flows op -> result.
                fwd.edge(fn_id, self._tensor_node(fwd, t, "f"))
                for s in fn.saved_tensors:
                    fwd.edge(self._tensor_node(fwd, s, "f"), fn_id)  # input tensor feeds the operation

    @staticmethod
    def _op_forward_name(fn: "Function") -> str:
        """Forward-facing operation name, e.g. MatmulBackward -> Matmul."""
        return type(fn).__name__.replace("Backward", "") or type(fn).__name__

    # ------------------------------------------------------------------ #
    # Backward computational graph cluster
    # ------------------------------------------------------------------ #
    def _build_backward(self, graph: graphviz.Digraph, out: Tensor) -> None:
        if out._grad_fn is None:
            return

        with graph.subgraph(name="cluster_backward") as bwd:  # pyright: ignore[reportUnknownMemberType, reportOptionalContextManager]
            bwd.attr(label="Backward Computational Graph", style="rounded",
                     color="#c05621", fontcolor="#c05621", fontsize="14")

            self._tensor_node(bwd, out, "t")
            for t in self._topo_order(out):
                fn = t._grad_fn
                assert fn is not None  # _topo_order only yields produced tensors
                fn_id = f"fn{id(fn)}"
                bwd.node(fn_id, type(fn).__name__, **_STYLE["op"])
                # Gradients flow the way they propagate -> from the result back to the op.
                bwd.edge(self._tensor_node(bwd, t, "t"), fn_id, label="grad")
                for s in fn.saved_tensors:
                    bwd.edge(fn_id, self._tensor_node(bwd, s, "t"))

    @staticmethod
    def _topo_order(root: Tensor) -> list[Tensor]:
        """Iterative post-order DFS over the autograd graph rooted at ``root``,
        yielding every tensor produced by an operation (i.e. with a
        ``_grad_fn``) exactly once, each after every tensor it (transitively)
        depends on - mirroring ``Tensor.__build_topo``.

        A tensor consumed by more than one downstream op (e.g. the
        hidden/cell state shared by every gate in an RNN/LSTM timestep) is
        marked "discovered" the moment it's first reached, so it is never
        re-queued for expansion by a later consumer; without that guard the
        whole upstream subgraph would be re-walked once per consumer,
        compounding into an exponential blow-up across timesteps. The DFS
        itself uses an explicit stack rather than function-call recursion,
        so it also doesn't blow Python's recursion limit on the long chains
        a many-timestep RNN/LSTM produces - the same failure mode the old
        recursive ``backward()`` had before it moved to a topo-sorted pass."""
        discovered: set[int] = {id(root)}
        order: list[Tensor] = []
        stack: list[tuple[Tensor, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                if node._grad_fn is not None:
                    order.append(node)
                continue
            stack.append((node, True))
            if node._grad_fn is not None:
                for child in node._grad_fn.saved_tensors:
                    if id(child) not in discovered:
                        discovered.add(id(child))
                        stack.append((child, False))
        return order

    def _tensor_node(self, g: graphviz.Digraph, t: Tensor, prefix: str) -> str:
        """Add (once) a tensor node labelled and coloured by its role."""
        node_id = f"{prefix}{id(t)}"
        role = self._tensor_role(t)
        g.node(node_id, self._tensor_label(role, t), **_ROLE_STYLE[role])
        return node_id

    @staticmethod
    def _tensor_role(t: Tensor) -> str:
        """One of: input, weights, bias, hidden.

        Parameters carry an explicit ``role`` (set by the layers); the synthetic
        forward input is tagged ``input``; everything else is a ``hidden`` tensor
        produced by an operation."""
        role = getattr(t, "role", None)
        if role in _ROLE_STYLE:
            return role
        return "hidden"

    @staticmethod
    def _tensor_label(role: str, t: Tensor) -> str:
        grad = "grad=set" if t.grad is not None else "grad=None"
        return f"{role}\nTensor {tuple(t.shape)}\n{grad}"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _run_forward(
        self, shape: tuple[int, ...]
    ) -> tuple[Tensor | None, list[tuple["Layer", tuple[int, ...], tuple[int, ...]]]]:
        """Run a real forward pass with an input of ``shape``, walking the
        model layer by layer so the real input/output shape of every layer
        is captured, then trigger backward so gradients show up on the
        picture.

        Returns the model output (or ``None`` if the model has no layers)
        and a list of ``(layer, in_shape, out_shape)`` triples in execution
        order.
        """
        layers = getattr(self.model, "layers", [self.model])
        if not layers:
            return None, []

        x = Tensor(np.ones(shape, dtype=np.float32))
        x.role = "input"

        layer_io: list[tuple["Layer", tuple[int, ...], tuple[int, ...]]] = []
        cur = x
        for layer in layers:
            in_shape = tuple(cur.shape)
            cur = layer(cur)
            layer_io.append((layer, in_shape, tuple(cur.shape)))

        out = cur
        # A backward pass populates .grad on every tensor in the graph so the
        # picture can report which tensors received gradients.
        try:
            out.sum().backward()
        except Exception:
            # Even if backward fails we still have a valid forward graph to draw.
            pass
        return out, layer_io
