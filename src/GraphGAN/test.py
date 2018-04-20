import numpy as np
import networkx as nx


app = "link_prediction"
# train_filename = "../../data/" + app + "/others" + "/CA-GrQc_undirected_train.txt"
train_filename = "../../data/" + app + "/others" + "/US_largest500_airportnetwork.txt"
# test_filename = "../../data/link_prediction/CA-GrQc_test.txt"
output_filename = "../../data/" + app + "/others" + "/US_largest500_airportnetwork.edgelist"
def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges

def read_edges(train_filename, test_filename, mode=""):
    """read the data from the file

    Args:
        train_filename:
        test_filename:
    Returns:
        train_edges: list, whose element is a list like [node1, node2]
        test_edges: list, whose element is a list like [node1, node2]
        linked_nodes: dict, dict, <node_id, linked_nodes_id>, store the neighbor nodes of every node
    """

    linked_nodes = {}
    train_edges = read_edges_from_file(train_filename)
    if test_filename != "":
        test_edges = read_edges_from_file(test_filename)
    else:
        test_edges = []
    start_nodes = set()
    end_nodes = set()

    for edge in train_edges:
        start_nodes.add(edge[0])
        end_nodes.add(edge[1])
        if linked_nodes.get(edge[0]) is None:
            linked_nodes[edge[0]] = []
        if linked_nodes.get(edge[1]) is None:
            linked_nodes[edge[1]] = []
        # undirected graph
        linked_nodes[edge[0]].append(edge[1])
        linked_nodes[edge[1]].append(edge[0])

    for edge in test_edges:
        start_nodes.add(edge[0])
        end_nodes.add(edge[1])
        if linked_nodes.get(edge[0]) is None:
            linked_nodes[edge[0]] = []
        if linked_nodes.get(edge[1]) is None:
            linked_nodes[edge[1]] = []

    if mode == "recommend":
        return len(start_nodes), len(end_nodes), linked_nodes
    else:
        return len(start_nodes.union(end_nodes)), linked_nodes


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def read_emd(filename, n_node, n_embed):
    """use the pretrain node embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
    node_embed = np.random.rand(n_node, n_embed)
    for line in lines:
        emd = line.split()
        node_embed[int(float(emd[0])), :] = str_list_to_float(emd[1:])

    return node_embed


# n_node, linked_nodes = read_edges(train_filename, test_filename)

# print(n_node)
# print(linked_nodes)

# n_node, linked_nodes = read_edges(train_filename, "")

# convert txt to edgelist
# G = nx.Graph()
#
# file = open(train_filename)
# # output = open(output_filename)
# for line in file.readlines():
#     edge = line.split()
#     if len(edge) >= 2:
#         # output.write(edge[0] + edge[1])
#         G.add_edge(int(edge[0]), int(edge[1]))

# print(G.neighbors(10))

# nx.write_edgelist(G, output_filename, data=False)
# random_state = np.random.randint(0, 100000)
# pretrain_emd_filename_d = "../../pre_train/" + app + "/p2p-Gnutella08.embeddings"
# pretrain_emd_filename_g = pretrain_emd_filename_d
# modes = ["dis", "gen"]
# emb_filenames = ["../../pre_train/" + app + "/CA-GrQc_" + modes[0] + "_" + str(random_state) + ".emb",
#                  "../../pre_train/" + app + "/CA-GrQc_" +  modes[1] + "_" + str(random_state) + ".emb"]
#
# n_embed = 50
# n_node = 5242
# node_embed_init_d = read_emd(filename=pretrain_emd_filename_d, n_node=n_node,
#                                         n_embed=n_embed)
# node_embed_init_g = read_emd(filename=pretrain_emd_filename_g, n_node=n_node,
#                                         n_embed=n_embed)
# print(node_embed_init_d.shape)


train_filename = "../../data/" + app + "/others" + "/CA-GrQc_undirected_train.txt"
test_filename = "../../data/link_prediction/CA-GrQc_test.txt"
test_neg_filename = "../../data/link_prediction/CA-GrQc_test_neg.txt"

print('Start counting...')

file = open(train_filename)
count = 0
for row in file.readlines():
    count += 1
print('train file size: ', count)

file = open(test_filename)
count = 0
for row in file.readlines():
    count += 1
print('train file size: ', count)

file = open(test_neg_filename)
count = 0
for row in file.readlines():
    count += 1
print('train file size: ', count)


train_filename = "../../data/" + app + "/others" + "/US_largest500_airportnetwork_undirected_train.txt"
test_filename = "../../data/link_prediction/US_largest500_airportnetwork_test.txt"
test_neg_filename = "../../data/link_prediction/US_largest500_airportnetwork_test_neg.txt"

print('Start counting...')

file = open(train_filename)
count = 0
for row in file.readlines():
    count += 1
print('train file size: ', count)

file = open(test_filename)
count = 0
for row in file.readlines():
    count += 1
print('test file size: ', count)

file = open(test_neg_filename)
count = 0
for row in file.readlines():
    count += 1
print('train file size: ', count)