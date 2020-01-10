import numpy as np
from collections import defaultdict
from torch.utils.data import dataset, dataloader

import globvar
from . import pickling


def get_children(nodes):
    node_dict = defaultdict(list)
    for node in nodes:
        if node.parent_id >= 0:
            node_dict[node.parent_id].append(node)
    return node_dict


def get_node_level(nodes, node):
    level = 0
    current_node = node
    while current_node.parent_id != -1:
        current_node = nodes[current_node.parent_id]
        level += 1
    return level


def get_nodes_by_level(nodes):
    node_dict = defaultdict(list)
    for node in nodes:
        node.level = get_node_level(nodes, node)
        node_dict[node.level].append(node)
    return node_dict


class Node:
    """ Basic clonal structure """

    def __init__(self, id, parent_id, genes, encoding, is_leaf):
        self.id = id
        self.parent_id = parent_id
        self.genes = genes
        self.encoding = encoding
        self.is_leaf = is_leaf
        self.level = None
        self.state = None

    def __repr__(self):
        return "<Node id=%s parent_id=%s level=%s>" % (self.id, self.parent_id, self.level)

    def __call__(self):
        return self.encoding


class Tree:
    """ A collection of Nodes indexed by their level. """

    def __init__(self, nodes, label):
        self._nodelist = nodes
        self.num_nodes = len(nodes)
        self.nodes = get_nodes_by_level(nodes)
        self.depth = max(self.nodes.keys())
        self.children = get_children(nodes)
        self.label = label
        self.root = self[0][0]

    def __iter__(self):
        for level in range(self.depth, -1, -1):
            for node in self.nodes[level]:
                yield node

    def __getitem__(self, index):
        return self.nodes[index]

    def __repr__(self):
        return "<Tree nodes=%s depth=%s>" % (self.num_nodes, self.depth)

    def __len__(self):
        return self.num_nodes


class Forest:
    """ A collection of trees. """

    def __init__(self, trees):
        self.trees = trees
        self.num_trees = len(trees)

    def __iter__(self):
        return iter(self.trees)

    def __repr__(self):
        return "<Forest trees=%s>" % self.num_trees

    def __len__(self):
        return self.num_trees

    def labels(self):
        return np.array([t.label for t in self])


if __name__ == '__main__':
    n1 = Node(0, -1, ["1"], [1], False)
    n2 = Node(1, 0, ["1"], [2], False)
    n3 = Node(2, 0, ["1"], [3], False)
    n4 = Node(3, 1, ["1"], [4], False)
    n5 = Node(4, 1, ["1"], [5], False)
    n6 = Node(5, 2, ["1"], [6], False)
    t1 = Tree([n1, n2, n3, n4, n5, n6], 1)
    t2 = Tree([n1, n2, n3, n4, n5, n6], 1)
    t3 = Tree([n1, n2, n3, n4, n5, n6], 1)

    f = Forest([t1, t2, t3])
