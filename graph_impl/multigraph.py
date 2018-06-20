from copy import copy
from copy import deepcopy

from graph_impl.graph import Graph


class Multigraph(Graph):
    def __init__(self, v_one=None, v_two=None, edges=None, bigraph=None, graph=None, ignore_inconsistent=True):
        if bigraph is not None:
            self = deepcopy(bigraph)
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
            super(Multigraph, self).__init__(edges=edges)
            self.incident_weighted = {}
            for node in self.incident:
                list_with_weights = []
                for elem in self.incident[node]:
                    list_with_weights.append([elem, 1])
                self.incident_weighted[node] = list_with_weights
            self._max_degree_intial = self.max_degree('degree')
            return

        if graph is None:
            super(Multigraph, self).__init__()
            self.node_coloring = {}
            self.incident = {}
        else:
            self.incident = graph.incident
            self.node_coloring = graph.node_coloring

        self.incident_weighted = {}
        for node in self.incident:
            list_with_weights = []
            for elem in self.incident[node]:
                list_with_weights.append([elem, 1])
            self.incident_weighted[node] = list_with_weights
        self._max_degree_intial = self.max_degree('degree')

    def add_edge(self, first_node, second_node):
        if first_node not in self.incident:
            self.add_node(first_node)

        if second_node not in self.incident:
            self.add_node(second_node)

        self.incident[first_node].append(second_node)
        self.incident[second_node].append(first_node)

    def remove_edge_from_incident_weighted_remove_strategy(self, first_node, second_node, weight):
        weighted_pair_remove_from_first = [second_node, weight]
        self.incident_weighted[first_node].remove(weighted_pair_remove_from_first)

        weighted_pair_remove_from_second = [first_node, weight]
        self.incident_weighted[second_node].remove(weighted_pair_remove_from_second)

        if len(self.incident_weighted[first_node]) == 0:
            del self.incident_weighted[first_node]

        if len(self.incident_weighted[second_node]) == 0:
            del self.incident_weighted[second_node]

    def remove_edge_from_incident_weighted_decrement_strategy(self, first_node, second_node, weight):
        new_weight = -1
        for weighted_pair in self.incident_weighted[first_node]:
            if weighted_pair[0] == second_node:
                if weighted_pair[1] - weight >= 0:
                    new_weight = weighted_pair[1] - weight
                    break

        if new_weight < 0:
            raise Exception()

        self.incident_weighted[first_node].remove(weighted_pair)
        if new_weight > 0:
            self.incident_weighted[first_node].append([second_node, new_weight])

        for weighted_pair in self.incident_weighted[second_node]:
            if weighted_pair[0] == first_node:
                if weighted_pair[1] - weight >= 0:
                    new_weight = weighted_pair[1] - weight
                    break

        if new_weight < 0:
            raise Exception()

        self.incident_weighted[second_node].remove(weighted_pair)
        if new_weight > 0:
            self.incident_weighted[second_node].append([first_node, new_weight])

        if len(self.incident_weighted[first_node]) == 0:
            del self.incident_weighted[first_node]

        if len(self.incident_weighted[second_node]) == 0:
            del self.incident_weighted[second_node]

    def add_edge_to_incident_weighted_append_strategy(self, first_node, second_node, weight):
        if first_node not in self.incident_weighted:
            self.incident_weighted[first_node] = [[second_node, weight]]
        else:
            self.incident_weighted[first_node].append([second_node, weight])

        if second_node not in self.incident_weighted:
            self.incident_weighted[second_node] = [[first_node, weight]]
        else:
            self.incident_weighted[second_node].append([first_node, weight])

    def add_edge_to_incident_weighted_increment_strategy(self, first_node, second_node, weight):
        if first_node not in self.incident_weighted:
            self.incident_weighted[first_node] = [[second_node, weight]]
        else:
            found = False
            for weighted_pair in self.incident_weighted[first_node]:
                if weighted_pair[0] == second_node:
                    weighted_pair[1] += weight
                    found = True
                    break
            if not found:
                self.incident_weighted[first_node].append([second_node, weight])

        if second_node not in self.incident_weighted:
            self.incident_weighted[second_node] = [[first_node, weight]]
        else:
            found = False
            for weighted_pair in self.incident_weighted[second_node]:
                if weighted_pair[0] == first_node:
                    weighted_pair[1] += weight
                    found = True
                    break
            if not found:
                self.incident_weighted[second_node].append([first_node, weight])

    def remove_node_from_incident_weighted(self, node):
        for elem_pair in self.incident_weighted[node]:
            node_incident = elem_pair[0]
            for cur in self.incident_weighted[node_incident]:
                if cur[0] == node:
                    self.incident_weighted[node_incident].remove(cur)
        del(self.incident_weighted[node])

    def euler_split(self, with_graph_copy):
        H1_edges = []
        H2_edges = []

        partitions = self.euler_partition(with_deepcopy=with_graph_copy)

        for partition in partitions:
            count = 0
            for edge in partition:
                if count % 2 == 0:
                    H1_edges.append(edge)
                else:
                    H2_edges.append(edge)
                count += 1

        H1_new = self.__class__()
        H2_new = self.__class__()

        for edge in H1_edges:
            H1_new.add_edge(edge[0], edge[1])
            H1_new.add_edge_to_incident_weighted_append_strategy(edge[0], edge[1], 1)
            H1_new.node_coloring = self.node_coloring

        for edge in H2_edges:
            H2_new.add_edge(edge[0], edge[1])
            H2_new.add_edge_to_incident_weighted_append_strategy(edge[0], edge[1], 1)
            H2_new.node_coloring = self.node_coloring
        return H1_new, H2_new

    def euler_partition(self, with_deepcopy):
        if with_deepcopy:
            graph_to_color_copy = deepcopy(self)
        else:
            graph_to_color_copy = copy(self)

        partitions = []
        queue = []
        odd_degrees = []
        even_degrees = []

        for node in graph_to_color_copy.incident:
            degree = graph_to_color_copy.degree(node)
            if degree % 2 == 0:
                even_degrees.append(node)
            else:
                odd_degrees.append(node)

        queue.extend(odd_degrees)
        queue.extend(even_degrees)

        while len(queue) != 0:
            first_node = queue[0]
            queue.remove(first_node)

            if graph_to_color_copy.degree(first_node) != 0:
                new_path = []
                current_node = first_node

                while graph_to_color_copy.degree(current_node) != 0:
                    processing_vertex_neighbors = graph_to_color_copy.neighbors(current_node)
                    next_node = processing_vertex_neighbors[0]
                    graph_to_color_copy.remove_edge(current_node, next_node)
                    new_path.append((current_node, next_node))
                    current_node = next_node

                partitions.append(new_path)
                if graph_to_color_copy.degree(first_node) != 0:
                    queue.append(first_node)

        return partitions
