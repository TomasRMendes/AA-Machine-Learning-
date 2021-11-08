import math
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


mat_test = np.loadtxt('TP1_test.tsv',delimiter='\t')
mat_train = np.loadtxt('TP1_train.tsv',delimiter='\t')


# shuffle TrainSet
data = shuffle(mat_train)
# shuffle TestSet
data = shuffle(mat_test)


# Ys contains the Classes (0 or 1)
Ys_test = mat_test[:,-1]
Ys_train = mat_train[:,-1]

# Xs contains the coordinates (x1, x2, ...)
Xs_test = mat_test[:,:4]
Xs_train = mat_train[:,:4]


# standartization of coordinates
means_train = np.mean(Xs_train,axis=0)
stdevs_train = np.std(Xs_train,axis=0)
# standardize(train)
Xs_train = (Xs_train-means_train)/stdevs_train
# standardize(test) based on train
Xs_test = (Xs_test-means_train)/stdevs_train


Xs_train_0 = []
Xs_train_1 = []


for i in range(len(Xs_train)): 
    if Ys_train[i]==0: 
        Xs_train_0.append(Xs_train[i,:])
    else: 
        Xs_train_1.append(Xs_train[i,:])



def calculate_confusion_matrix(kdes, test_set):
    confusion_matrix = [0,# Real and Predicted Real
                        0,# Fake but Predicted Real
                        0,# Real but Predicted Fake
                        0]# Fake and Predicted Fake
    
    scores = []
    scores.append(kdes[0].score_samples(test_set))
    scores.append(kdes[1].score_samples(test_set))
    
    
    for i in range(len(test_set)): 
        c = 0
        if scores[1][i]>scores[0][i]:
            c = 1
        if Ys_test[i]==c and c == 0:
            confusion_matrix[0] += 1
        elif Ys_test[i]==c and c == 1: 
            confusion_matrix[3] += 1
        elif Ys_test[i]!=c and c == 1:
            confusion_matrix[1] += 1
        else: 
            confusion_matrix[2] += 1
    #kde.score_samples(X_t)
    
    return confusion_matrix
    
def print_confusion_matrix(mat): 
    print("Predictions \tReal\tFake")
    print("Real\t\t\t"+str(mat[0])+"\t\t"+str(mat[1]))
    print("Fake\t\t\t"+str(mat[2])+"\t\t"+str(mat[3]))
    

    
    


kde = [None,None]
kde[0] = KernelDensity(kernel='gaussian', bandwidth=1)
kde[0].fit(Xs_train_0)
kde[1] = KernelDensity(kernel='gaussian', bandwidth=1)
kde[1].fit(Xs_train_1)

mat = calculate_confusion_matrix(kde, Xs_test)
print_confusion_matrix(mat)

"""
1.
Naïve Bayes classifier using Kernel Density Estimation


2.
Gaussian Naïve Bayes classifier


3.
Support Vector Machine with a Gaussian radial basis function
"""































