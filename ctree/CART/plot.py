
import networkx as nx
import matplotlib.pyplot as plt


def get_subnodes(node):
    ''' Get the subnodes of a node
    '''
    l = node.left
    r = node.right
    return l, r


def compute_next_layer(layer, G):
    ''' Given a layer of a tree, find all subnodes one layer down
        and add them to the graph G.
    '''
    next_layer = []
    for node in layer:
        try:
            cl, cr = get_subnodes(node)
        except KeyError: # This node is a leaf
            continue
            
        G.add_edge(node, cl)
        G.add_edge(node, cr)
        
        next_layer.append(cl)
        next_layer.append(cr)
        
    return next_layer, G



def parse_tree_as_nx(node, max_eval = 1000):
    ''' Parse a decision tree into a 
        networkx graph
    '''
    G = nx.Graph()
    G.add_node(node)
    
    n = 0
    nxt = [node]
    while True:
        nxt, G = compute_next_layer(nxt, G)
        
        n += 1
        if len(nxt) == 0:
            break
        if n > max_eval:
            raise ValueError('Tree to deep. Reached max_eval before finishing.')        
    return G


def plot_nx_tree(G, labels = False):
    ''' Plot '''
    plt.figure(figsize = (12,12))
    pos=nx.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=False, arrows=False)
    return plt


def plot_partition(partition, labels = False):
    Gtree = parse_tree_as_nx(partition.dtree)
    return plot_nx_tree(Gtree, labels)