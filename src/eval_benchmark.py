import evaluation.eval_link_prediction as elp
import argparse


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
    conf.test_filename = 'data/%s/%s_test.txt' % (conf.app, conf.data)
    conf.test_neg_filename = 'data/%s/%s_test_neg.txt' % (conf.app, conf.data)
    conf.result_filename = 'results/%s/%s_accuracy.txt' % (conf.app, conf.data)
    conf.emb_filename = 'embedding/%s/%s.emb' % (conf.model, conf.data)
    eval_test(conf)
