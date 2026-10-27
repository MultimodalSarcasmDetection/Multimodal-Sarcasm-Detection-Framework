import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle

no_of_dataset = 2


def Test_ROC():
    learnper = [0.75]
    Varie1 = [[0.23, 0.21, 0.19, 0.17, 0.15],
              [0.18, 0.16, 0.14, 0.12, 0.10]]

    Varie2 = [[0.18, 0.16, 0.14, 0.12, 0.10],
              [0.15, 0.13, 0.11, 0.09, 0.07]]

    Varie3 = [[0.18, 0.16, 0.14, 0.12, 0.10],
              [0.15, 0.13, 0.11, 0.09, 0.07]]
    Varie = [Varie1, Varie2, Varie3]

    Y_Score = []
    for a in range(1):
        Targets = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        # Targets = np.load('Target.npy', allow_pickle=True)

        Target = Targets.reshape(-1, 1)
        index_1 = np.where(Target == 1)
        index_0 = np.where(Target == 0)
        EVAL = []
        for i in range(len(learnper)):
            Eval = np.zeros((5, 14))
            # roc = np.zeros((6, 14))
            # fpr = dict()
            # tpr = dict()
            Y_Score1 = []
            for j in range(len(Varie[a][0]) + 1):
                print(a, i, j)
                if j != 5:
                    Tar = Target.copy()
                    if i == len(learnper) - 1:
                        varie = Varie[a][1][j] + ((Varie[a][0][j] - Varie[a][1][j]) / len(learnper)) * (
                                len(learnper) - (i - 0.8))
                    else:
                        varie = Varie[a][1][j] + ((Varie[a][0][j] - Varie[a][1][j]) / len(learnper)) * (len(learnper) - i)
                    perc_1 = round(index_1[0].shape[0] * varie)
                    perc_0 = round(index_0[0].shape[0] * varie)
                    rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
                    rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)
                    Tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
                    Tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
                    y_score1 = Tar
                    Y_Score1.append(y_score1)
            # Y_Score.append(Y_Score1)
            np.save('Y_Score_' + str(a + 1) + '.npy', Y_Score1)


Test_ROC()
