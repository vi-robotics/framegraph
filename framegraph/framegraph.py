from typing import Union
import numpy as np
import igraph

from framegraph.pose import Pose


class FrameGraph():
    """A frame graph which supports rapid relative transform computation.
    """

    def __init__(self, base_node_name: str = "world"):
        self._base_node_name = base_node_name
        self._graph = igraph.Graph(directed=True)
        self._graph.add_vertex(name=self._base_node_name)

    def add_node(self,
                 node_name: str):
        graph_names = self._graph.vs["name"]
        if node_name in graph_names:
            raise ValueError(f"node_name {node_name} is already present in the"
                             " graph.")
        self._graph.add_vertex(name=node_name)

    def set_edge(self,
                 source: Union[int, str],
                 target: Union[int, str],
                 transform: Pose = None,
                 timestamp: float = None):
        self._graph.add_edge

    def update_edge(self,
                    source: Union[int, str],
                    target: Union[int, str],
                    transform: Pose = None,
                    timestamp: float = None):
        pass

    def update_edge_lambda(self,
                           source: Union[int, str],
                           target: Union[int, str],
                           timestamp: float = None,
                           *lambda_args,
                           **lambda_kwargs):
        pass

    def get_relative_transform(self,
                               source: Union[int, str],
                               target: Union[int, str],
                               timestamps: Union[float, np.ndarray] = None,
                               reverse: bool = False
                               ):
        pass
