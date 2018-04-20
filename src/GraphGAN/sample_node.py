import numpy as np

input_filename = "../../data/link_prediction/others/US_largest500_airportnetwork.txt"
test_filename = "../../data/link_prediction/US_largest500_airportnetwork_test.txt"
train_filename = "../../data/link_prediction/others/US_largest500_airportnetwork_undirected_train.txt"

nodeMap = {}

sample_num = 298
total_num = 2980

sample_idx = list()

# generate sample number
i = 0
while i < sample_num:
    rnd = np.random.randint(0, total_num)
    if rnd not in sample_idx:
        sample_idx.append(rnd)
        i += 1

i = 0
inputFile = open(input_filename)
testFile = open(test_filename, "w+")
trainFile = open(train_filename, "w+")
for row in inputFile.readlines():
    edge = row.split()
    if i not in sample_idx:
        trainFile.write(edge[0] + ' ' + edge[1] + '\n')
    else:
        testFile.write(edge[0] + ' ' + edge[1] + '\n')
    i += 1
    edge = row.split()


