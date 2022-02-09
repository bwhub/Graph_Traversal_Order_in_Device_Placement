"""This module contains classes that are related to graph traversal order.

Available class:
- GraphTraversalOrder: contains functions specifies orders of travesing a DAG.
"""

import networkx as nx
from networkx.algorithms.dag import topological_sort

class GraphTraversalOrder(object):
    """This class contains functions that defines different DAG traversal order.

    Available functions:
    - alpha_sort(G): Returns a list of nodes in alphabetically sorted order based on the names of nodes;
    - topological_sort(G): Returns a generator of nodes in topologically sorted order;
    - lexicographical_topological_sort(G): Returns a generator of nodes in lexicographically topologically sorted order;
    - reverse_topological_sort(G): Returns a list of nodes in reversed topologically sorted order;
    - dfs_preorder_sort(G): Returns a list of nodes in a depth-first-search pre-ordering;
    - dfs_postorder_sort(G): Returns a list of nodes in a depth-first-search post-ordering;
    - bfs_sort(G): Returns a list of nodes in a breadth-first-search.
    
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def alpha_sort(G):
        """Returns a list of nodes in alphabetically sorted order based on the names of nodes.
        
        """
        return sorted(list(G.nodes))

    @staticmethod
    def topological_sort(G):
        """Returns a list of nodes in topologically sorted order.
        
        This function uses nx.topological_sort() as implementation.
        """
        return list(nx.topological_sort(G))

    @staticmethod
    def lexicographical_topological_sort(G):
        """Returns a generator of nodes in lexicographically topologically sorted order.
        
        This function uses nx.lexicographical_topological_sort() as implementation.
        It will choose zero degree nodes based on the alphabetical order.
        By default, the "smallest" one will be chosen.
        """
        return list(nx.lexicographical_topological_sort(G))


    @staticmethod
    def reverse_topological_sort(G):
        """Returns a list of nodes in reversed topologically sorted order.
        Also known as post-order.
        
        """
        topo_list = list(nx.topological_sort(G))
        return list(reversed(topo_list))
    
    @staticmethod
    def dfs_preorder_sort(G):
        """Returns a list of nodes in a depth-first-search pre-ordering.
        A source is chosen arbitratily and repeatedly until all components in the graph are searched.
        """
        return list(nx.dfs_preorder_nodes(G))

    @staticmethod
    def dfs_postorder_sort(G):
        """Returns a list of nodes in a depth-first-search post-ordering.
        A source is chosen arbitratily and repeatedly until all components in the graph are searched.
        """
        return list(nx.dfs_postorder_nodes(G))

    @staticmethod
    def bfs_sort(G):
        """Returns a list of nodes in a breadth-first-search.
        A source is chosen arbitratily and repeatedly until all components in the graph are searched.
        """
        zero_indegree_nodes = sorted([v for v, d in G.in_degree() if d == 0])
        
        node_list_with_duplicates = []
        for source in zero_indegree_nodes:
            node_list_with_duplicates.append(source)
            node_list_with_duplicates += [v for _, v in nx.bfs_edges(G, source)]
        
        visited = set()
        visited_add = visited.add
        return [node for node in node_list_with_duplicates if not (node in visited or visited_add(node)) ]
