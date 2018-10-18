import numpy as np

bf0 = np.load('buffer_0.npy')
bf1 = np.load('buffer_1.npy')

for i, v in enumerate(bf0):

    equal = (bf1[i][2] == bf0[i][2])
    equal = equal and (bf1[i][1] == bf0[i][1]) and (np.sum(abs(bf1[i][0] - bf0[i][0])) < 0.000001)
    if not equal:
        print(i)
        print(equal)
        print(bf1[i][2], bf0[2])
    # print(bf1[i] == bf0[i])
