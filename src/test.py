data = "CA-HepPh"
test_filename = "../../data/link_prediction/%s_test2.txt" % data
outF = open(test_filename, 'w')
count = 0
with open("../../data/link_prediction/others/%s.edgelist" % data) as f:
    while count < 11848:
        row = f.readline()
        outF.write(row)
        count += 1