import matplotlib.pyplot as plt
import numpy as np


def plot_emb():
    deepwalk = [0.550, 0.607, 0.614, 0.587, 0.587, 0.597]
    line_first = [0.640, 0.708, 0.751, 0.761, 0.775, 0.761]
    line_second = [0.681, 0.644, 0.661, 0.674, 0.691, 0.697]
    graphGAN = [0.664, 0.728, 0.771, 0.791, 0.838, 0.879]
    node2vec = [0.825, 0.825, 0.838, 0.828, 0.818, 0.838]
    emb = []
    i = 8
    while i <= 256:
        emb.append(i)
        i *= 2

    plt.semilogx(emb, deepwalk, 'gs-', basex=2, linewidth=1.5, label='deepwalk')
    plt.semilogx(emb, line_first, 'r^-', basex=2, linewidth=1.5, label='line-1st order')
    plt.semilogx(emb, line_second, 'bo-', basex=2, linewidth=1.5, label='line-2nd order')
    plt.semilogx(emb, graphGAN, 'mv-', basex=2, linewidth=1.5, label='GraphGAN')
    plt.semilogx(emb, node2vec, 'yv-', basex=2, linewidth=1.5, label='node2vec')
    plt.xlabel('d')
    plt.ylabel('Acc')
    plt.grid()
    plt.legend()
    plt.title('US largest500 airport network')
    plt.show()
    # plt.savefig('US_airport.png')


def plot_emb2():
    deepwalk =[0.639, 0.680, 0.752, 0.784]
    graphGAN = [0.656, 0.723, 0.808, 0.849]
    line_first = [0.714, 0.707, 0.680, 0.642]
    line_second = [0.681, 0.716, 0.718, 0.705]
    node2vec = [0.845, 0.851, 0.859, 0.868]
    emb = []
    i = 16
    while i <= 128:
        emb.append(i)
        i *= 2
    plt.semilogx(emb, deepwalk, 'gs-', basex=2, linewidth=1.5, label='deepwalk')
    plt.semilogx(emb, line_first, 'r^-', basex=2, linewidth=1.5, label='line-1st order')
    plt.semilogx(emb, line_second, 'bo-', basex=2, linewidth=1.5, label='line-2nd order')
    plt.semilogx(emb, graphGAN, 'mv-', basex=2, linewidth=1.5, label='GraphGAN')
    plt.semilogx(emb, node2vec, 'yv-', basex=2, linewidth=1.5, label='node2vec')
    plt.xlabel('d')
    plt.ylabel('Acc')
    plt.grid()
    plt.legend()
    plt.title('CA-GrQc')
    plt.show()
    # plt.savefig('CA_GrQc.png')


def plot_iter():
    emb32_dis = [0.679, 0.663, 0.651, 0.645, 0.642, 0.642]
    emb32_gen = [0.680, 0.693, 0.700, 0.710, 0.718, 0.723]
    emb64_dis = [0.750, 0.729, 0.719, 0.716, 0.720, 0.727]
    emb64_gen = [0.75, 0.765, 0.779, 0.789, 0.801, 0.808]
    emb128_dis = [0.781, 0.753, 0.747, 0.758, 0.774, 0.788]
    emb128_gen = [0.783, 0.809, 0.826, 0.834, 0.844, 0.849]
    i = np.arange(0, 6)
    plt.plot(i, emb32_dis, 'gs--', linewidth=1.5, label='emb32_dis')
    plt.plot(i, emb32_gen, 'gs-', linewidth=1.5, label='emb32_gen')
    plt.plot(i, emb64_dis, 'r^--', linewidth=1.5, label='emb64_dis')
    plt.plot(i, emb64_gen, 'r^-', linewidth=1.5, label='emb64_gen')
    plt.plot(i, emb128_dis, 'bo--', linewidth=1.5, label='emb128_dis')
    plt.plot(i, emb128_gen, 'bo-', linewidth=1.5, label='emb128_gen')
    plt.xlabel('iteration')
    plt.ylabel('Acc')
    plt.grid()
    plt.legend()
    # plt.title('CA-GrQc')
    # plt.show()
    plt.savefig('CA_GrQc_GraphGAN.png')


def plot_iter2():
    emb32_dis = [0.614, 0.620, 0.624, 0.624, 0.627, 0.647, 0.667, 0.684, 0.697, 0.711, 0.724]
    emb32_gen = [0.614, 0.651, 0.681, 0.711, 0.734, 0.765, 0.791, 0.812, 0.825, 0.832, 0.842]
    emb64_dis = [0.610, 0.617, 0.617, 0.620, 0.634, 0.640, 0.654, 0.664, 0.674, 0.681, 0.694]
    emb64_gen = [0.610, 0.644, 0.677, 0.704, 0.731, 0.761, 0.775, 0.791, 0.798, 0.805, 0.818]
    emb128_dis = [0.587, 0.597, 0.607, 0.630, 0.644, 0.671, 0.708, 0.738, 0.761, 0.791, 0.812]
    emb128_gen = [0.587, 0.657, 0.714, 0.761, 0.802, 0.832, 0.852, 0.869, 0.879, 0.879, 0.885]
    i = np.arange(0, 11)
    plt.plot(i, emb32_dis, 'gs--', linewidth=1.5, label='emb32_dis')
    plt.plot(i, emb32_gen, 'gs-', linewidth=1.5, label='emb32_gen')
    plt.plot(i, emb64_dis, 'r^--', linewidth=1.5, label='emb64_dis')
    plt.plot(i, emb64_gen, 'r^-', linewidth=1.5, label='emb64_gen')
    plt.plot(i, emb128_dis, 'bo--', linewidth=1.5, label='emb128_dis')
    plt.plot(i, emb128_gen, 'bo-', linewidth=1.5, label='emb128_gen')
    plt.xlabel('iteration')
    plt.ylabel('Acc')
    plt.grid()
    plt.legend()
    plt.title('US largest500 airport network')
    plt.show()
    # plt.savefig('US_airport_GraphGAN.png')


plot_emb2()
# plot_iter2()


