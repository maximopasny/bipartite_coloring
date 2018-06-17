import copy
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

import os

import sys


class Graph:

    """
    Creates an instance of `Graph`.

    Args:
        graph (obj:Graph, optional)              - basic graph
        edges (obj:list of (int, int), optional) - edges
    """

    def __init__(self, edges=None, graph=None):
        # map key - (v1, v2) value - map of params: 'weight', 'color'

        # weight = {(v, u) : int}
        # color  = {(v, u) : int}

        # see: property

        if graph is not None:
            self = copy.deepcopy(graph)
            pass

        node_coloring_copy = {}
        if not hasattr(self, 'node_coloring'):
            self.node_coloring = {}
        else:
            node_coloring_copy = self.node_coloring.copy()

        if not hasattr(self, 'edge_coloring'):
            self.edge_coloring = {}

        if not hasattr(self, '_max_degree'):
            self._max_degree = 0

        if not hasattr(self, 'incident'):
            self.incident = {}

        if edges is not None:
            self.edges = edges
            for edge in edges:
                self.add_edge(edge[0], edge[1])

        if node_coloring_copy != {}:
            self.node_coloring = node_coloring_copy

    def add_node(self, node):
        self.incident[node] = []
        self.node_coloring = {}

    def add_node_with_set(self, node, set):
        self.incident[node] = []
        self.node_coloring[node] = set

    def add_edge(self, first_node, second_node):
        if first_node not in self.incident:
            self.add_node(first_node)

        if second_node not in self.incident:
            self.add_node(second_node)

        if second_node not in self.incident[first_node]:
            self.incident[first_node].append(second_node)

        if first_node not in self.incident[second_node]:
            self.incident[second_node].append(first_node)

        self.node_coloring = {}

    def remove_edge(self, first_node, second_node):
        if second_node in self.incident[first_node]:
            self.incident[first_node].remove(second_node)
        if first_node in self.incident[second_node]:
            self.incident[second_node].remove(first_node)

        # self.node_coloring = {}

    def remove_node(self, node):
        neighbors = self.incident[node]
        for neighbor in neighbors:
            self.incident[neighbor].remove(node)

        self.incident.pop(node)
        self.node_coloring.pop(node)

    def neighbors(self, node):
        return self.incident[node]

    def incident_edges(self, node):
        return [(node, neighbor) for neighbor in self.incident[node]]

    def degree(self, node):
        degree = len(self.neighbors(node))
        return degree

    def max_degree(self, out='node'):
        if not hasattr(self, 'incident'):
            return
        elif len(self.incident) == 0:
            if out == 'degree':
                return 0
            else:
                return
        max_degree_node = max(self.incident, key=lambda node: len(self.incident[node]))
        if out == 'node':
            return max_degree_node
        if out == 'degree':
            max_degree = len(self.incident[max_degree_node])
            self._max_degree = max_degree
            return max_degree
        else:
            return

    def max_degree_vertices(self):
        max_degree_vertices = set()
        max_degree = self.max_degree('degree')
        for node in self.incident:
            if self.degree(node) == max_degree:
                max_degree_vertices.add(node)
        return max_degree_vertices

    def get_edges(self):
        result = []
        for node_one in self.incident:
            for node_two in self.incident[node_one]:
                if (node_two, node_one) not in result:
                    result.append((node_one, node_two))

        return result

    def union(self, graph):
        copy_self = copy.deepcopy(self)
        for node1 in graph.incident:
            for node2 in graph.incident[node1]:
                copy_self.add_edge(node1, node2)
        return copy_self

    # alpha = 0.5
    def draw_with_node_edge_coloring(self, filename=None, edge_coloring=None):
        nxgraph = nx.Graph()
        red_nodes = []
        black_nodes = []
        for node in self.node_coloring:
            if self.node_coloring[node] == 1:
                black_nodes.append(node)
            else:
                red_nodes.append(node)

        nxgraph.add_nodes_from(red_nodes, bipartite=0)
        nxgraph.add_nodes_from(black_nodes, bipartite=1)

        edges = []
        for node in self.incident:
            for another_node in self.incident[node]:
                edges.append((node, another_node))

        nxgraph.add_edges_from(edges)
        position = nx.circular_layout(nxgraph)
        nx.draw_networkx_nodes(nxgraph, pos=position, node_color='Black', nodelist=black_nodes, node_size=300)
        nx.draw_networkx_nodes(nxgraph, pos=position, node_color='Red', nodelist=red_nodes, node_size=300)
        colors = ['Purple', 'Orange', 'Yellow', 'Green', 'Brown',
                  'Pink', 'Gray', 'Aqua', 'Azure',
                  'Bazaar', 'Blood']
        it = 0
        for color in edge_coloring:
            nx.draw_networkx_edges(G=nxgraph, edgelist=edge_coloring[color], edge_color=colors[it], pos=position,
                                   width=4)
            it += 1

        black_edges=[]
        for edge in edges:
            found = False
            for color in edge_coloring:
                if edge in edge_coloring[color]:
                    found = True
            if not found:
                black_edges.append(edge)
        nx.draw_networkx_edges(G=nxgraph, edgelist=black_edges, edge_color='black', pos=position,
                               width=4)
        plt.axis('off')
        os.makedirs(sys.path[0] + '/results', exist_ok=True)
        if filename is not None:
            plt.savefig(filename + " " + str(datetime.now()) + ".png")  # save as png
        else:
            plt.savefig("results/" + "coloring " + str(datetime.now()) + ".png")

        #plt.show()
        plt.clf()

    def draw_graph_simple(self, edge_color, filename, pos=None):
        G_nx = nx.Graph()
        red_nodes = []
        black_nodes = []
        for node in self.node_coloring:
            if self.node_coloring[node] == 1:
                black_nodes.append(node)
            else:
                red_nodes.append(node)

        G_nx.add_nodes_from(red_nodes, bipartite=0)
        G_nx.add_nodes_from(black_nodes, bipartite=1)

        edges = []
        for node in self.incident:
            for another_node in self.incident[node]:
                edges.append((node, another_node))

        G_nx.add_edges_from(edges)
        if pos is None:
            pos = nx.spring_layout(G_nx)

        nx.draw_networkx_nodes(G=G_nx, pos=pos, node_color='Gray', nodelist=black_nodes)
        nx.draw_networkx_nodes(G=G_nx, pos=pos, node_color='Red', nodelist=red_nodes)
        nx.draw_networkx_edges(G=G_nx, edgelist=edges, edge_color=edge_color, pos=pos, width=4)
        plt.axis('off')
        plt.savefig("results/" + filename)
        plt.clf()

        return pos

    def has_no_edges(self):
        has_edges = False
        for edge in self.incident:
            if len(self.incident[edge]) > 0:
                has_edges = True
                break
        return not has_edges