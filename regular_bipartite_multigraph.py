import logging
import random
from bipartite_base import BipartiteBase
from multigraph import Multigraph
from math import floor
from math import log2
from copy import deepcopy


class RegularBipartiteGraph(Multigraph, BipartiteBase):
    i = 0

    def __init__(self, v_one=None, v_two=None,
                 edges=None, bigraph=None, graph=None,
                 ignore_inconsistent=True):
        super(RegularBipartiteGraph, self).__init__(v_one=v_one,
                                                    v_two=v_two,
                                                    edges=edges,
                                                    bigraph=bigraph,
                                                    graph=graph,
                                                    ignore_inconsistent=ignore_inconsistent)
        self.weight_to_graph_mapping = {}
        if edges is not None or (v_one is not None and v_two is not None) or graph is not None:
            self.make_regular()
            self.reduce_edges()
            for weight in self.weight_to_graph_mapping:
                if weight == 0:
                    continue
                subgraph = self.weight_to_graph_mapping[weight]
                for node in subgraph.incident_weighted:
                    weighted_incident_list = subgraph.incident_weighted[node]
                    if node in self.incident_weighted:
                        self.incident_weighted[node] += weighted_incident_list
                    else:
                        self.incident_weighted[node] = weighted_incident_list

    def reduce_edges(self):
        max_degree = self.max_degree('degree')
        weights_to_process = [2**i for i in range(1 + floor(log2(max_degree)))]
        g_zero_weight = RegularBipartiteGraph()
        weight_to_graph_mapping = {0: g_zero_weight}
        self.weight_to_graph_mapping = weight_to_graph_mapping
        itergraph = self.copy()
        for weight in weights_to_process:
            if len(itergraph.incident_weighted) > 0:
                itergraph = self.procede_weight(itergraph, weight_to_graph_mapping, g_zero_weight, weight)

    @staticmethod
    def get_root_vertex_for_circut_processing(incident_weighted):
        for candidate_to_be_the_root in incident_weighted:
            for weighted_pair in incident_weighted[candidate_to_be_the_root]:
                if weighted_pair[1] > 0:
                    return candidate_to_be_the_root

    def make_regular(self):
        max_dergee = self.max_degree('degree')
        set_one = set()
        set_two = set()
        new_vertex_index = 0
        for node in self.incident:
            if node > new_vertex_index:
                new_vertex_index = node
            if self.node_coloring[node] == 0:
                set_one.add(node)
            else:
                set_two.add(node)

        set_size_delta = len(set_one) - len(set_two)
        new_vertex_index += 1
        if set_size_delta >= 0:
            set_bigger, set_less = set_one, set_two
        else:
            set_bigger, set_less = set_two, set_one

        set_to_extend_with = set()
        if set_size_delta == 0:
            pass
        elif set_size_delta < 0:
            while set_size_delta != 0:
                self.add_node_with_set(new_vertex_index, 0)
                set_to_extend_with.add(new_vertex_index)
                set_less.add(new_vertex_index)
                new_vertex_index += 1
                set_size_delta += 1
        else:
            while set_size_delta != 0:
                self.add_node_with_set(new_vertex_index, 1)
                set_to_extend_with.add(new_vertex_index)
                set_less.add(new_vertex_index)
                new_vertex_index += 1
                set_size_delta -= 1

        self.set_one, self.set_two = set_bigger, set_less
        #TODO: save time from degree access by creating a set of available
        for set_bigger_elem in set_bigger:
            degree = self.degree(set_bigger_elem)
            while degree < max_dergee:
                for set_less_elem in set_less:
                    if self.degree(set_less_elem) < max_dergee:
                        self.add_edge(set_bigger_elem, set_less_elem)
                        degree += 1
                        if degree == max_dergee:
                            break

        self.incident_weighted = {}
        for node in self.incident:
            list_to_append_weighted = []
            for neighbor in self.incident[node]:
                list_to_append_weighted.append([neighbor, 1])
            self.incident_weighted[node] = list_to_append_weighted

    def procede_weight(self, itergraph, weight_to_graph_mapping, g_zero_weight, weight):
        g_double, g_same = itergraph.process_circuts(weight, g_zero_weight)
        weight_to_graph_mapping[weight] = g_same
        return g_double

    def process_circuts(self, weight, g_zero_weight):
        g_double = RegularBipartiteGraph()
        g_same = RegularBipartiteGraph()
        visited = {node: False for node in self.incident_weighted}
        node = self.get_root_vertex_for_circut_processing(self.incident_weighted)
        visited[node] = True
        stack = [node]
        parent = None

        while len(self.incident_weighted) > 0:
            #if current node has no incident neighbors
            if node not in self.incident_weighted or len(self.incident_weighted[node]) == 0:
                if node in self.incident_weighted:
                    if len(self.incident_weighted[node]) == 0:
                        del self.incident_weighted[node]

                if len(self.incident_weighted) == 0:
                    return g_double, g_same

                node = random.choice(list(self.incident_weighted))
                visited = {node: False for node in self.incident_weighted}
                stack = [node]
                visited[node] = True
                parent = None
            ###########################################
            #if current node has only parent as incident
            only_parent_incident = True
            for elem in self.incident_weighted[node]:
                if elem[0] != parent:
                    only_parent_incident = False

            if only_parent_incident:
                for _ in range(len(self.incident_weighted[node])):
                    g_same.add_edge_to_incident_weighted_append_strategy(node, parent, weight)
                self.remove_node_from_incident_weighted(node)
                stack.pop()
                node = parent
                if len(stack) >= 2:
                    parent = stack[len(stack) - 2]
                else:
                    parent = None
                continue
            ###########################################

            #choosing neighbor
            neighbor_node = parent
            temp = 0
            while neighbor_node == parent:
                neighbor_weighted_pair = self.incident_weighted[node][temp]
                neighbor_node = neighbor_weighted_pair[0]
                temp += 1
            ###########################################

            if visited[neighbor_node] is True:
                # cycle found, need to store it to temp list and append to global storage
                cur_vertex_cycle = [neighbor_node]
                cur_node = stack.pop()
                while cur_node != neighbor_node:
                    cur_vertex_cycle.append(cur_node)
                    cur_node = stack.pop()
                cur_vertex_cycle.append(neighbor_node)
                if len(stack) > 0:
                    parent = stack[len(stack) - 1]
                else:
                    parent = None

                cur_vertex_cycle_len = len(cur_vertex_cycle)
                for i in range(cur_vertex_cycle_len - 1):
                    if i != 0 and i != cur_vertex_cycle_len - 1:
                        intermediate_path_node = cur_vertex_cycle[i]
                        visited[intermediate_path_node] = False
                    node_one = cur_vertex_cycle[i]
                    node_two = cur_vertex_cycle[i+1]
                    self.remove_edge_from_incident_weighted_remove_strategy(node_one, node_two, weight)
                    if i % 2 == 0:
                        #g_zero_weight.add_edge_to_incident_weighted_append_strategy(node_one, node_two, 0)
                        pass
                    else:
                        g_double.add_edge_to_incident_weighted_append_strategy(node_one, node_two, weight * 2)

                stack.append(neighbor_node)
                node = neighbor_node
                continue
            else:
                visited[neighbor_node] = True
                stack.append(neighbor_node)
                parent = node
                node = neighbor_node

        return g_double, g_same

    def get_matchings(self):
        visited = {node: False for node in self.incident_weighted}
        node = self.get_root_vertex_for_circut_processing(self.incident_weighted)
        visited[node] = True
        stack = [node]
        parent = None
        final_matchings = []

        while self.have_to_continue_matchings_search():
            # if current node has no incident neighbors
            if node not in self.incident_weighted or len(self.incident_weighted[node]) == 0:
                if node in self.incident_weighted:
                    if len(self.incident_weighted[node]) == 0:
                        del self.incident_weighted[node]

                if len(self.incident_weighted) == 0:
                    return

                node = random.choice(list(self.incident_weighted))
                visited = {node: False for node in self.incident_weighted}
                stack = [node]
                visited[node] = True
                parent = None
            ###########################################

            # if current node has only parent as incident
            only_parent_incident = True
            for elem in self.incident_weighted[node]:
                if elem[0] != parent:
                    only_parent_incident = False

            if only_parent_incident:
                final_matchings.append([node, parent])
                final_matchings.append([parent, node])
                self.remove_node_from_incident_weighted(node)
                stack.pop()
                node = parent
                if len(stack) >= 2:
                    parent = stack[len(stack) - 2]
                else:
                    parent = None
                continue
            ###########################################

            # choosing neighbor
            neighbor_node = parent
            temp = 0
            while neighbor_node == parent:
                neighbor_weighted_pair = self.incident_weighted[node][temp]
                neighbor_node = neighbor_weighted_pair[0]
                temp += 1
            ###########################################

            if visited[neighbor_node] is True:
                cur_vertex_cycle = [neighbor_node]
                cur_node = stack.pop()
                while cur_node != neighbor_node:
                    cur_vertex_cycle.append(cur_node)
                    cur_node = stack.pop()
                cur_vertex_cycle.append(neighbor_node)
                if len(stack) > 0:
                    parent = stack[len(stack) - 1]
                else:
                    parent = None

                matching_to_increment = []
                matching_to_decrement = []
                even_index_mathcing = []
                odd_index_matching = []
                even_odd_weights_delta = 0
                min_weight = 999999999

                cur_vertex_cycle_len = len(cur_vertex_cycle)
                for i in range(cur_vertex_cycle_len - 1):
                    if i != 0 and i != cur_vertex_cycle_len - 1:
                        intermediate_path_node = cur_vertex_cycle[i]
                        visited[intermediate_path_node] = False
                    node_one = cur_vertex_cycle[i]
                    node_two = cur_vertex_cycle[i + 1]
                    edge_weight = -1
                    for neighbor in self.incident_weighted[node_one]:
                        if neighbor[0] == node_two:
                            edge_weight = neighbor[1]

                    if edge_weight < min_weight:
                        min_weight = edge_weight

                    if i % 2 == 0:
                        even_index_mathcing.append([node_one, node_two])
                        even_odd_weights_delta += edge_weight
                    else:
                        odd_index_matching.append([node_one, node_two])
                        even_odd_weights_delta -= edge_weight

                if even_odd_weights_delta >= 0:
                    matching_to_increment = even_index_mathcing
                    matching_to_decrement = odd_index_matching
                else:
                    matching_to_increment = odd_index_matching
                    matching_to_decrement = even_index_mathcing

                for edge in matching_to_decrement:
                    self.remove_edge_from_incident_weighted_decrement_strategy(edge[0], edge[1], min_weight)

                for edge in matching_to_increment:
                    self.add_edge_to_incident_weighted_increment_strategy(edge[0], edge[1], min_weight)

                stack.append(neighbor_node)
                node = neighbor_node
                continue
            else:
                visited[neighbor_node] = True
                stack.append(neighbor_node)
                parent = node
                node = neighbor_node

        return final_matchings

    def have_to_continue_matchings_search(self):
        for node in self.incident_weighted:
            if len(self.incident_weighted[node]) > 1:
                return True
        return False

    def test_matchings(self):
        for node in self.incident_weighted:
            assert len(self.incident_weighted[node]) == 1
            assert self.incident_weighted[node][0][1] == self._max_degree_intial
            assert self.incident_weighted[node][0][0] in self.incident[node]

        for node in self.incident:
            assert node in self.incident_weighted

    def process_vertex_set_lack_degrees(self, less_than_half_dict, less_than_max_dict, max_degree):
        has_nodes_to_merge = len(less_than_half_dict) > 1
        keys_from_vertex_lack_degrees = less_than_half_dict.keys()
        while has_nodes_to_merge:
            node_one = random.choice(keys_from_vertex_lack_degrees)
            keys_from_vertex_lack_degrees.remove(node_one)
            node_two = random.choice(keys_from_vertex_lack_degrees)
            node_one_incident = self.incident.pop(node_one)
            del keys_from_vertex_lack_degrees[node_one]
            for node_one_neighbor in node_one_incident:
                self.incident[node_two].append(node_one_neighbor)
            node_two_new_degree = len(self.incident[node_two])
            keys_from_vertex_lack_degrees[node_two] = node_two_new_degree
            if node_two_new_degree > max_degree / 2:
                del keys_from_vertex_lack_degrees[node_two]
                if node_two_new_degree < max_degree:
                    less_than_max_dict[node_two] = node_two_new_degree
            if len(keys_from_vertex_lack_degrees) <= 1:
                has_nodes_to_merge = False

    def assert_regular(self):
        max_dergee = self.max_degree('degree')
        for node in self.incident:
            if self.degree(node) != max_dergee:
                #logging.error('degree is ' + str(self.degree(node)))
                raise Exception()
        #logging.debug("regularity ok!")

    def assert_g_same_g_double_keep_degree(self, g_same, g_double):
        vertices_sum = {x: 0 for x in self.incident}
        for node in g_same.incident_weighted:
            weights = [cur[1] for cur in g_same.incident_weighted[node]]
            appendix = sum(weights)
            vertices_sum[node] += appendix

        for node in g_double.incident_weighted:
            weights = [cur[1] for cur in g_double.incident_weighted[node]]
            appendix = sum(weights)
            vertices_sum[node] += appendix

        result1 = []
        for node in vertices_sum:
            if vertices_sum[node] != self.max_degree('degree'):
                result1.append((node, vertices_sum[node]))

        #logging.debug(result1)
        assert len(result1) == 0

    def assert_after_iteration_reduced_keep_degree(self, g_double):
        vertices_sum = {x: 0 for x in self.incident}
        for weight in self.weight_to_graph_mapping:
            cur_graph = self.weight_to_graph_mapping[weight]
            for node in cur_graph.incident_weighted:
                weights = [temp[1] for temp in cur_graph.incident_weighted[node]]
                appendix = sum(weights)
                vertices_sum[node] += appendix

        for node in g_double.incident_weighted:
            weights = [cur[1] for cur in g_double.incident_weighted[node]]
            appendix = sum(weights)
            vertices_sum[node] += appendix

        result1 = []
        for node in vertices_sum:
            if vertices_sum[node] != self.max_degree('degree'):
                result1.append((node, vertices_sum[node]))

        #logging.debug(result1)
        assert len(result1) == 0
        #logging.debug("assert ok")

    def assert_final_reduced_keep_degree(self):
        vertices_sum = {x: 0 for x in self.incident}
        for weight in self.weight_to_graph_mapping:
            cur_graph = self.weight_to_graph_mapping[weight]
            for node in cur_graph.incident_weighted:
                weights = [temp[1] for temp in cur_graph.incident_weighted[node]]
                appendix = sum(weights)
                vertices_sum[node] += appendix

        result1 = []
        for node in vertices_sum:
            if vertices_sum[node] != self.max_degree('degree'):
                result1.append((node, vertices_sum[node]))

        #logging.debug(result1)
        assert len(result1) == 0
        #logging.debug("assert ok")

    def make_regular_with_merge(self):
        max_dergee = self.max_degree('degree')
        v1_lack_degree = {}
        v2_lack_degree = {}
        v1_less_than_max = {}
        v2_less_than_max = {}
        for node in self.incident:
            degree = len(self.incident)
            if len(self.incident) <= max_dergee / 2:
                if self.node_coloring[node] == 0:
                    v1_lack_degree[node] = degree
                else:
                    v2_lack_degree[node] = degree
            elif len(self.incident) < max_dergee:
                if self.node_coloring[node] == 0:
                    v1_less_than_max[node] = degree
                else:
                    v2_less_than_max[node] = degree
        has_nodes_to_merge_v1 = len(v1_lack_degree) > 1
        if has_nodes_to_merge_v1:
            self.process_vertex_set_lack_degrees(v1_lack_degree, v1_less_than_max, max_dergee)
        has_nodes_to_merge_v2 = len(v2_lack_degree) > 1
        if has_nodes_to_merge_v2:
            self.process_vertex_set_lack_degrees(v2_lack_degree, v2_less_than_max, max_dergee)

        has_vertices_to_increase_degree = (len(v1_less_than_max) > 0 and len(v2_less_than_max) > 0)
        if has_vertices_to_increase_degree:
            raise Exception()

    def find_arbitrary_matching(self):
        not_covered_v1 = {node for node in self.set_one}
        not_covered_v2 = {node for node in self.set_two}
        m_arbitrary = []
        for v1_node in self.set_one:
            for v2_node in self.set_two:
                if v2_node in self.incident[v1_node]:
                    m_arbitrary.add((v1_node, v2_node))
                    not_covered_v1.remove(v1_node)
                    not_covered_v2.remove(v2_node)
                    break

        for v1_not_covered_node in not_covered_v1:
            found = False
            for v2_not_covered_node in not_covered_v2:
                m_arbitrary.add((v1_not_covered_node, v2_not_covered_node))
                found = True
                break
            if found:
                not_covered_v2.remove(v2_not_covered_node)

        return m_arbitrary

    def find_matchings_alon(self):
        max_degree = self.max_degree('degree')
        t = floor(log2(max_degree))
        alpha = 2**t / max_degree
        beta = 2**t - max_degree*alpha
        m = self.find_arbitrary_matching()

        for node in self.incident_weighted:
            for neighbor_weighted_pair in self.incident_weighted[node]:
                neighbor_weighted_pair[1] *= alpha

        for edge in m:
            v, u = edge[0], edge[1]
            self.incident_weighted[v][u] += beta
            self.incident_weighted[u][v] += beta



