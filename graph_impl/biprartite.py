import copy

from graph_impl.graph import Graph
from graph_impl.bipartite_base import BipartiteMixin


class Bipartite(Graph, BipartiteMixin):
    i = 0

    def __init__(self, v_one=None, v_two=None, edges=None, bigraph=None, graph=None, ignore_inconsistent=True):
        if bigraph is not None:
            self = copy.deepcopy(bigraph)
            return

        if v_one is not None and v_two is not None:

            is_edges = edges is not None
            is_graph = graph is not None
            if is_edges and is_graph:
                raise Exception('edges and graph cannot be not None simultaneously')

            edges = edges if is_edges else graph.get_edges() if is_graph else []

            for edge in edges:
                v0_in_one = edge[0] in v_one
                v0_in_two = edge[0] in v_two
                v1_in_one = edge[1] in v_one
                v1_in_two = edge[1] in v_two

                if not ((v0_in_one ^ v0_in_two) and (v1_in_one ^ v1_in_two) and (v0_in_one ^ v1_in_one)):
                    if ignore_inconsistent:
                        edges.remove(edge)
                    else:
                        raise Exception('inconsistent edges for bipartite graph initialization')

            node_coloring = {v: 0 for v in v_one}
            node_coloring.update({v: 1 for v in v_two})
            self.node_coloring = node_coloring
            self.incident = {}
            super().__init__(edges=edges)
            return

        if graph is None:
            super().__init__()
            self.node_coloring = {}
            self.incident = {}