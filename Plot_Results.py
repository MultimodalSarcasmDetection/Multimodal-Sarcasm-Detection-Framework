from itertools import cycle
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn import metrics
import matplotlib.patheffects as patheffects
from matplotlib.patches import FancyBboxPatch

def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plot_results():
    eval1 = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Algorithm = ['TERMS', 'CSA-GAF-ABiRNN', 'NGO-GAF-ABiRNN', 'MFO-GAF-ABiRNN', 'HOA-GAF-ABiRNN', 'EHO-GAF-ABiRNN']
    Classifier = ['TERMS', 'mBERT-BiLSTM-MHA', 'QNNs', 'GRU', 'Bi-RNN', 'EHO-GAF-ABiRNN']
    for i in range(eval1.shape[0]):
        for m in range(eval1.shape[1]):
            value1 = eval1[i, m, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Terms[:8])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value1[j, :8])
            print('-------------------------------------------------- Dataset -', str(i + 1), 'Fold - ', str(m + 1),
                  ' -  Algorithm Comparison',
                  '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Terms[:8])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value1[j + 5, :8])
            print('-------------------------------------------------- Dataset -', str(i + 1), 'Fold - ', str(m + 1),
                  ' -  Method Comparison',
                  '--------------------------------------------------')
            print(Table)


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CSA-GAF-ABiRNN', 'NGO-GAF-ABiRNN', 'MFO-GAF-ABiRNN', 'HOA-GAF-ABiRNN', 'EHO-GAF-ABiRNN']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Dataset = ['Dataaet', 'Dataset', 'Dataset']
    for i in range(1):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report for ', Dataset[i],
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color=[0, 0.5, 0.5], linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color=[0.5, 0, 0.5], linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['mBERT-BiLSTM-MHA', 'QNNs', 'GRU', 'Bi-RNN', 'EHO-GAF-ABiRNN']
    for a in range(1):  # For 2 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_results_optimizer():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 14, 16, 8, 9, 10]
    Algorithm = ['TERMS', 'CSA-GAF-ABiRNN', 'NGO-GAF-ABiRNN', 'MFO-GAF-ABiRNN', 'HOA-GAF-ABiRNN', 'EHO-GAF-ABiRNN']
    Classifier = ['TERMS', 'mBERT-BiLSTM-MHA', 'QNNs', 'GRU', 'Bi-RNN', 'EHO-GAF-ABiRNN']
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]

    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
            Optimizer = ['Adam', 'SGD', 'RMSProp', 'Adadelta', 'AdaGrad']

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.7, 0.7])
            ax.set_facecolor("#f0f0f0")
            fig.canvas.manager.set_window_title('Optimizer vs ' + Terms[Graph_Term[j]])
            plt.plot(Optimizer, Graph[:, 0], color='m', linewidth=3, marker='o', markersize=10,
                     label=Algorithm[1])
            plt.plot(Optimizer, Graph[:, 1], color='y', linewidth=3, marker='o', markersize=10,
                     label=Algorithm[2])
            plt.plot(Optimizer, Graph[:, 2], color='g', linewidth=3, marker='h', markersize=10,
                     label=Algorithm[3])
            plt.plot(Optimizer, Graph[:, 3], color='r', linewidth=3, marker='h', markersize=10,
                     label=Algorithm[4])
            plt.plot(Optimizer, Graph[:, 4], color='k', linewidth=3, marker='*', markersize=16,
                     label=Algorithm[5])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19),
                       ncol=3, fancybox=True, shadow=True)

            plt.xticks(Optimizer, ('Adam', 'SGD', 'RMSProp', 'Ada-delta', 'AdaGrad'), fontname="Arial", fontsize=12, fontweight='bold',
                       color='#35530a')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold',
                       color='#35530a')
            plt.xlabel('Optimizer', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.grid(color='c', linestyle='-', linewidth=2)

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset-%s-%s-Algorithm.png" % (i + 1, Terms[Graph_Term[j]])
            # path = "./Results/%s_Alg_line.png" % (Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()

            # Plot bars and add labels above bars
            # fig = plt.figure(figsize=(10, 6))
            # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            # Eval = eval
            #
            # X = np.arange(Eval.shape[2] - 5)  # X positions for categories
            # barWidth = 0.15
            # colors = ['#9b5de5', '#f15bb5', '#fee440', '#00bbf9', '#00f5d4']
            # Classifier = ["DES", "AES", "ECC", "Elgamal Algorithm", "E-2-TF"]
            #
            # for m in range(Eval.shape[1]):
            #     barz = ax.bar(X + (m * barWidth), Graph[:, m], color=colors[m], width=barWidth, edgecolor='#032b43', label=Classifier[m])
            #     for i, bar in enumerate(barz):
            #         if m == i:  # Show the classifier name only for one bar in each case
            #             height = bar.get_height()
            #
            #             if m == 3:
            #                 # Display classifier name
            #                 # ax.text(bar.get_x() + bar.get_width()/0.6, height + (0.29 * height), Classifier[m], ha='center', va='bottom',
            #                 ax.text(bar.get_x() + bar.get_width() / 0.25, (np.max(Graph[:, m])) + 0.05, Classifier[m], ha='center', va='bottom',
            #                         fontweight='heavy', fontsize=8, color='k')
            #                 # path_effects=[patheffects.withStroke(linewidth=1, foreground='black')])
            #             else:
            #                 ax.text(bar.get_x() + bar.get_width() / 0.5, (np.max(Graph[:, m])) + 0.05, Classifier[m], ha='center', va='bottom',
            #                         fontweight='heavy', fontsize=8, color='k')
            #                 # path_effects=[patheffects.withStroke(linewidth=1, foreground='black')])
            #
            #             # Add a connecting line and underline
            #             text_x = bar.get_x() + bar.get_width() / 0.99
            #             text_y = height + (0.30 * height)
            #             bar_mid_x = bar.get_x() + bar.get_width() / 2
            #
            #             # Line connecting text and bar
            #             ax.plot([bar_mid_x, text_x], [height, (np.max(Graph[:, m])) + 0.05], color=colors[m], lw=0.5)
            #
            #             # Colored underline
            #             ax.add_patch(FancyBboxPatch(
            #                 (text_x + 0.0018, (np.max(Graph[:, m])) + 0.04),  # Start point of underline
            #                 len(Classifier[m]) / 17,  # Width of underline
            #                 0.010,  # Height of underline
            #                 boxstyle="square,pad=0",
            #                 linewidth=0,
            #                 edgecolor=colors[m],
            #                 facecolor=colors[m],
            #             ))
            #
            # # Set x-axis labels
            # plt.xticks(X + (((len(Classifier)) * barWidth) / 2.30), ('1', '2', '3', '4', '5'))
            #
            # # # Legend with markers
            # # dot_markers = [
            # #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8)
            # #     for c in colors
            # # ]
            # # plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.13), fontsize=8, frameon=False, ncol=len(Classifier))
            #
            # # Hide top and right spines
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # plt.xlabel('Case')
            # plt.ylabel('CPA')
            # path1 = "./Results/CPA_Encryption_Met_%s.png" % (str(i + 1))
            # plt.savefig(path1)
            # plt.show()
            #
            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.7, 0.7])
            ax.set_facecolor("#f0f0f0")
            fig.canvas.manager.set_window_title('Optimizer vs ' + Terms[Graph_Term[j]])
            plt.plot(Optimizer, Graph[:, 5], color='m', linewidth=3, marker='o', markersize=10,
                     label=Classifier[1])
            plt.plot(Optimizer, Graph[:, 6], color='y', linewidth=3, marker='o', markersize=10,
                     label=Classifier[2])
            plt.plot(Optimizer, Graph[:, 7], color='g', linewidth=3, marker='h', markersize=10,
                     label=Classifier[3])
            plt.plot(Optimizer, Graph[:, 8], color='r', linewidth=3, marker='h', markersize=10,
                     label=Classifier[4])
            plt.plot(Optimizer, Graph[:, 9], color='k', linewidth=3, marker='*', markersize=16,
                     label=Classifier[5])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19),
                       ncol=3, fancybox=True, shadow=True)

            plt.xticks(Optimizer, ('Adam', 'SGD', 'RMSProp', 'Ada-delta', 'AdaGrad'), fontname="Arial", fontsize=12, fontweight='bold',
                       color='#35530a')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold',
                       color='#35530a')
            plt.xlabel('Optimizer', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.grid(color='c', linestyle='-', linewidth=2)

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset-%s-%s-Method.png" % (i + 1, Terms[Graph_Term[j]])
            # path = "./Results/%s_Alg_line.png" % (Terms[Grapuih_Term[j]])
            plt.savefig(path)
            plt.show()


def Image_Results():
    Dataset = ['Periocular', 'Iris Dataset', 'Sclera Dataset']
    for m in range(3):
        Images = np.load('Images_'+str(m+1)+'.npy', allow_pickle=True)
        Enhanced_Images = np.load('Enhanced_Images_'+str(m+1)+'.npy', allow_pickle=True)
        for n in range(10): # len(Images)
            cv.imwrite('./Results/Image_Results/'+ Dataset[m] + 'original-' + str(n + 1) + '.png', Images[n])
            cv.imwrite('./Results/Image_Results/'+ Dataset[m] + 'Enhanced-' + str(n + 1) + '.png', Enhanced_Images[n])


if __name__ == '__main__':
    # Image_Results()
    plot_results_optimizer()
    plot_results()
    plotConvResults()
    Plot_ROC_Curve()
