from typing import Union, Callable, Dict, Any, Tuple
import operator
from functools import reduce
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jit, jacrev
import igraph

from framegraph.pose import Pose


class FrameGraph():
    """A frame graph which supports rapid relative transform computation.
    """

    def __init__(self, base_node_name: str = "world"):
        self._base_node_name = base_node_name
        self._graph = igraph.Graph(directed=True)
        self._graph.add_vertex(name=self._base_node_name)
        # A dictionary of cached gradients
        self._grad_map: Dict[Tuple[int, int], Any] = {}

    def add_node(self,
                 node_name: str):
        graph_names = self._graph.vs["name"]
        if node_name in graph_names:
            raise ValueError(f"node_name {node_name} is already present in the"
                             " graph.")
        self._graph.add_vertex(name=node_name)

    def add_edge(self,
                 source: Union[int, str],
                 target: Union[int, str],
                 transform_callback: Callable[..., jnp.ndarray],
                 params: np.ndarray):
        if not isinstance(params, np.ndarray):
            raise ValueError("params must be an ndarray")
        if params.ndim != 1:
            raise ValueError("params must be a 1d array")
        attrs: Dict[str, Any] = {
            "params": params,
            "transform_callback": transform_callback
        }
        rev_attrs: Dict[str, Any] = {
            "params": params,
            "transform_callback": jit(lambda x: jnp.linalg.inv(transform_callback(x))),
        }
        self._graph.add_edge(source, target, **attrs)
        self._graph.add_edge(target, source, **rev_attrs)

    def get_params(self, source: Union[int, str],
                   target: Union[int, str]):
        source_id = self._node_to_id(source)
        target_id = self._node_to_id(target)
        paths = self._graph.get_shortest_paths(source_id, target_id)
        path = paths[0]
        eids = self._graph.get_eids(path=path, directed=False)
        edges = self._graph.es[eids]
        all_params = [e["params"] for e in edges]
        named_edge_nodes = [
            (self._graph.vs[e.source]["name"],
             self._graph.vs[e.target]["name"]) for e in edges]

        return named_edge_nodes, all_params

    def register_transform(self,
                           source: Union[int, str],
                           target: Union[int, str]):
        source_id = self._node_to_id(source)
        target_id = self._node_to_id(target)

        paths = self._graph.get_shortest_paths(source_id, target_id)
        if len(np.squeeze(paths)) == 0:
            raise ValueError("No path exists from source to target.")

        # Get an arbitrary path
        path = paths[0]
        eids = self._graph.get_eids(path=path, directed=False)
        edges = self._graph.es[eids]
        callbacks = [e["transform_callback"] for e in edges]
        all_params = [e["params"] for e in edges]
        params_slices = np.cumsum(
            np.array([len(p.reshape((-1,))) for p in all_params]))

        @jit
        def forward_multiply(params):
            mats = []
            for i, callback in enumerate(callbacks):
                if i == 0:
                    p = params[:params_slices[i]]
                else:
                    p = params[params_slices[i - 1]:params_slices[i]]
                mats.append(callback(p))
            return reduce(operator.matmul, mats)

        # We should use reverse mode unless there are more parameters than
        # elements in a transformation matrix
        if params_slices[-1] < 16:
            jac = jit(jacrev(forward_multiply))
        else:
            jac = jit(jacfwd(forward_multiply))
        # We need to call the functions to get them to jit compile
        ps = np.concatenate(all_params)
        forward_multiply(ps)
        jac(ps)

        key = (source_id, target_id)
        self._grad_map[key] = {"trans_func": forward_multiply,
                               "trans_jac": jac,
                               "path": path}

    def get_relative_transform(self,
                               source: Union[int, str],
                               target: Union[int, str],
                               ret_jac: bool = False
                               ):
        source_id = self._node_to_id(source)
        target_id = self._node_to_id(target)

        paths = self._graph.get_shortest_paths(source_id, target_id)
        if len(np.squeeze(paths)) == 0:
            raise ValueError("No path exists from source to target.")

        key = (source_id, target_id)
        if key not in self._grad_map:
            self.register_transform(source, target)
        cached_grad = self._grad_map[key]

        # Get an arbitrary path
        path = paths[0]

        # If a different path is taken, then we should re-register the gradient
        if not np.allclose(path, cached_grad["path"]):
            self.register_transform(source, target)
            cached_grad = self._grad_map[key]

        eids = self._graph.get_eids(path=path)
        edges = self._graph.es[eids]
        ps = np.concatenate([e["params"] for e in edges])
        if not ret_jac:
            return cached_grad["trans_func"](ps)
        return cached_grad["trans_func"](ps), cached_grad["trans_jac"](ps)

    def _node_to_id(self, node: Union[str, int]):
        if isinstance(node, str):
            return self._graph.vs.find(name=node).index
        else:
            return node
