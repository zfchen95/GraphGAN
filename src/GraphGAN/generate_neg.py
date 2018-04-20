import numpy as np


test_filename = "../../data/link_prediction/US_largest500_airportnetwork_test.txt"
output_filename = "../../data/link_prediction/US_largest500_airportnetwork_test_neg.txt"
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