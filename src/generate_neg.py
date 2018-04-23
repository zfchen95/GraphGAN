import numpy as np

"""
Generates undirected graph for training, testing
Training: 90% of the edge pairs
Edgelist: Same content as training dataset in edgelist format
Testing: 10% of the edge pairs
Negative: Same number with testing file of edge pairs that does not exist in the graph 
"""
data = "CA-HepPh"
input_filename = "../data/link_prediction/others/%s.txt" % data
edgelist_filename = "../data/link_prediction/others/%s.edgelist" % data
train_filename = "../data/link_prediction/others/%s_undirected_train.txt" % data
test_filename = "../data/link_prediction/%s_test.txt" % data
output_filename = "../data/link_prediction/%s_test_neg.txt" % data

nodeMap = {}
for i in range(500):
    nodeMap[i] = list()
file = open(test_filename)
for row in file.readlines():
    edge = row.split()
    nodeMap[int(edge[0])].append(int(edge[1]))
    nodeMap[int(edge[1])].append(int(edge[0]))
print('finish building map')


output = open(output_filename, "w+")
i = 0
while i < 298:
    node1 = np.random.randint(0, 500)
    node2 = np.random.randint(0, 500)
    if node1 != node2 and node1 not in nodeMap[node2]:
        i += 1
        output.write(str(node1) + ' ' + str(node2) +'\n')
print('finish writing')
output.close()