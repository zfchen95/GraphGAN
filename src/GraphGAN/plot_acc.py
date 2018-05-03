import matplotlib.pyplot as plt
import numpy as np

deepwalk = [0.550, 0.607, 0.614, 0.587, 0.587, 0.597]
line_first = [0.640, 0.708, 0.751, 0.761, 0.775, 0.761]
line_second = [0.681, 0.644, 0.661, 0.674, 0.691, 0.697]
emb = []
i = 8
while i <= 256:
    emb.append(i)
    i *= 2

plt.semilogx(emb, deepwalk, 'gs-', basex=2, linewidth=1.5, label='deepwalk')
plt.semilogx(emb, line_first, 'r^-', basex=2, linewidth=1.5, label='line-1st order')
plt.semilogx(emb, line_second, 'bo-', basex=2, linewidth=1.5, label='line-2nd order')
plt.xlabel('d')
plt.ylabel('Precision')
plt.grid()
plt.legend()
plt.title('US largest500 airport network')
# plt.show()
plt.savefig('US_airport.png')