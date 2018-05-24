import evaluation.eval_link_prediction as elp
import argparse
import utils
import numpy as np
from sklearn.metrics import f1_score

import os

"""
commands:
deepwalk --input example_graphs/ca-GrQc.edgelist --output ../GraphGAN/embedding/deepwalk/ca-GrQc.emb --format edgelist --representation-size 256 --walk-length 80 --window-size 10 --number-walks 10

deepwalk --input example_graphs/US_largest500_airportnetwork.edgelist --output ../GraphGAN/embedding/deepwalk/US_largest500_airportnetwork_256.emb --format edgelist --representation-size 256 --walk-length 80 --window-size 10 --number-walks 10

deepwalk --input example_graphs/CA-HepPh.edgelist --output ../GraphGAN/embedding/deepwalk/CA-HepPh.emb --format edgelist --representation-size 128 --walk-length 80 --window-size 10 --number-walks 10

deepwalk --input example_graphs/ca-GrQc.edgelist --output ../GraphGAN/embedding/ca-GrQc/ca-GrQc_8.emb --format edgelist --representation-size 8 --walk-length 80 --window-size 10 --number-walks 10

line:
python line.py --graph_file ca-GrQc --output ca-GrQc_second_8 --dimensions 8 --proximity second-order 

python line.py --dimensions 16 --output US_largest500_airportnetwork_first_16 --proximity first-order 

node2vec
python src/main.py --input graph/ca-GrQc.edgelist --output ../GraphGAN/embedding/node2vec/ca-GrQc.emb --dimensions 128 --window-size 10 --walk-length 80 --num-walks 1

python src/main.py --input graph/US_largest500_airportnetwork.edgelist --output ../GraphGAN/embedding/US_largest500_airportnetwork/US_largest500_airportnetwork_p05_q2_128.emb --dimensions 128 --window-size 10 --walk-length 80 --num-walks 10 --p 0.5 --q 2

struc2vec
python src/main.py --input graph/ca-GrQc.edgelist --output emb/ca-GrQc.emb --dimensions 128 --window-size 10 --walk-length 80 --num-walks 10 --OPT1 True --OPT2 True --OPT3 True

python graph_gan.py

python src/eval_model.py --app link_prediction --n_embed 128 --data CA-HepPh --n_node 12006 --model deepwalk

python src/eval_model.py --app link_prediction --n_embed 256 --data ca-GrQc --n_node 5241 --model node2vec

python src/eval_model.py --app link_prediction --n_embed 256 --data US_largest500_airportnetwork --n_node 500 --model node2vec
"""


def eval_test(config):
    """do the evaluation when training

    :return:
    """
    results = []
    if config.app == "link_prediction":
        LPE = elp.LinkPredictEval(config.emb_filename, config.test_filename, config.test_neg_filename,
                                  config.n_node, config.n_embed)
        result = LPE.eval_link_prediction()
        results.append(config.model + ":" + str(result) + "\n")

    with open(config.result_filename, mode="a+") as f:
        f.writelines(results)

    test_edges = utils.read_edges_from_file(config.test_filename)
    test_edges_neg = utils.read_edges_from_file(config.test_neg_filename)
    test_edges.extend(test_edges_neg)
    emd = utils.read_emd(config.emb_filename, n_node=config.n_node, n_embed=config.n_embed)
    score_res = []
    for i in range(len(test_edges)):
        score_res.append(np.dot(emd[test_edges[i][0]], emd[test_edges[i][1]]))
    test_label = np.array(score_res)
    bar = np.median(test_label)
    ind_pos = test_label >= bar
    ind_neg = test_label < bar
    test_label[ind_pos] = 1
    test_label[ind_neg] = 0
    true_label = np.zeros(test_label.shape)
    true_label[0:len(true_label) // 2] = 1
    f1 = f1_score(true_label, test_label, average='macro')
    result = config.model + ":" + str(f1) + "\n"
    print(result)
    with open(config.result_filename_f1, mode="a+") as f:
        f.writelines(result)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating embeddings")
    parser.add_argument('--app', default='link_prediction')
    parser.add_argument('--data', default='US_largest500_airportnetwork')
    parser.add_argument('--model', default='GraphGAN_gen')
    # parser.add_argument('--emb_filename', default='embedding/GraphGAN_gen/%s.emb' % parser.model)
    # parser.add_argument('--test_filename', default='data/link_prediction/US_largest500_airportnetwork_test.txt')
    # parser.add_argument('--test_neg_filename', default='data/link_prediction/US_largest500_airportnetwork_test_neg.txt')
    # parser.add_argument('--result_filename', default='results/link_prediction/US_largest500_airportnetwork_accuracy.txt')
    parser.add_argument('--n_node', type=int, default=500)
    parser.add_argument('--n_embed', type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    conf = parse_args()

    # conf.data = 'ca-GrQc'
    # conf.data = 'CA-HepPh'
    conf.emb_filename = 'embedding/%s/%s_%d.emb' % (conf.model, conf.data, conf.n_embed)
    conf.model = '%s_%d' % (conf.data, conf.n_embed)
    emb = int(conf.model.split('_')[-1])
    conf.n_embed = emb

    conf.test_filename = 'data/%s/%s_test.txt' % (conf.app, conf.data)
    conf.test_neg_filename = 'data/%s/%s_test_neg.txt' % (conf.app, conf.data)
    conf.result_filename = 'results/%s/%s_accuracy.txt' % (conf.app, conf.data)
    conf.result_filename_f1 = 'results/%s/%s_macrof1.txt' % (conf.app, conf.data)

    eval_test(conf)

    # conf.emb_filename = 'embedding/%s/' % conf.model
    #
    # for filename in os.listdir(conf.emb_filename):
    #     tmp = filename.replace('.emb', '')
    #
    #     try:
    #         emb = int(tmp.split('_')[-1])
    #         conf.n_embed = emb
    #         conf.emb_filename = 'embedding/%s/%s' % (conf.data, filename)
    #         conf.model = tmp
    #         eval_test(conf)
    #     except ValueError:
    #         print(tmp.split('_')[-1])
