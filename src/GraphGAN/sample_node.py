import numpy as np

"""
Generates undirected graph for training, testing
Training: 90% of the edge pairs
Edgelist: Same content as training dataset in edgelist format
Testing: 10% of the edge pairs
Negative: Same number with testing file of edge pairs that does not exist in the graph 
"""
data = "CA-HepPh"
input_filename = "../../data/link_prediction/others/%s.txt" % data
undirected_filename = "../../data/link_prediction/others/%s_undirected.txt" % data
edgelist_filename = "../../data/link_prediction/others/%s.edgelist" % data
train_filename = "../../data/link_prediction/others/%s_undirected_train.txt" % data
test_filename = "../../data/link_prediction/%s_test.txt" % data
neg_filename = "../../data/link_prediction/%s_test_neg.txt" % data


directed = False


def generate_edges():
    total_edges = 0
    nodeMap = {}
    nodeList = list()
    with open(undirected_filename) as f:
        for row in f.readlines():
            edge = row.split()
            if edge[0] not in nodeMap:
                nodeMap[edge[0]] = list()
                nodeList.append(edge[0])
            nodeMap[edge[0]].append(edge[1])
            if edge[1] not in nodeMap:
                nodeMap[edge[1]] = list()
                nodeList.append(edge[1])
            nodeMap[edge[1]].append(edge[0])
            total_edges += 1
    f.close()
    print("total edges:", total_edges)
    sample_edges = int(total_edges * 0.1)
    sample_idx = list()
    # generate testing edges index
    i = 0
    while i < sample_edges:
        rnd = np.random.randint(0, total_edges)
        if rnd not in sample_idx:
            sample_idx.append(rnd)
            i += 1

    i = 0
    undirectedFile = open(undirected_filename)
    trainFile = open(train_filename, "w+")
    edgelistFile = open(edgelist_filename, "w+")
    testFile = open(test_filename, "w+")
    for row in undirectedFile.readlines():
        edge = row.split()
        if i not in sample_idx:
            trainFile.write('%d %d \n' % (int(edge[0]), int(edge[1])))
            edgelistFile.write('%d %d \n' % (int(edge[0]), int(edge[1])))
        else:
            testFile.write('%d %d \n' % (int(edge[0]), int(edge[1])))
        i += 1
    print("Positive edges generation done")
    # generate negative edges
    generate_neg_edges(sample_edges, nodeMap, nodeList)


def generate_neg_edges(neg_edges, nodeMap, nodeList):
    negFile = open(neg_filename, "w+")
    node_num = len(nodeList)
    print(node_num)
    i = 0
    while i < neg_edges:
        rnd1 = np.random.randint(0, node_num)
        rnd2 = np.random.randint(0, node_num)
        if rnd1 == rnd2:
            continue
        if nodeList[rnd1] in nodeMap and nodeList[rnd2] in nodeMap[nodeList[rnd1]]:
            continue
        negFile.write('%d %d \n' % (int(nodeList[rnd1]), int(nodeList[rnd2])))
        i += 1


if directed:
    with open(undirected_filename, 'w+') as outF:
        with open(input_filename) as f:
            for row in f.readlines():
                edge = row.split()
                if int(edge[0]) < int(edge[1]):
                    outF.write('%d %d \n' % (int(edge[0]), int(edge[1])))
else:
    generate_edges()



