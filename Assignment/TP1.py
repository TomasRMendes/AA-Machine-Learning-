import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.naive_bayes import  GaussianNB
from sklearn import svm
import math


class OwnNaiveBaseClassifier:
    
    kde = []
    
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        
    
    def fit(self, Xs_train_set,Ys_train_set):
                
        # split train data according to their class
        Xs_train_class = []
        for i in range(int(max(Ys_train_set))+1):
            Xs_train_class.append([])
        for i in range(len(Ys_train_set)):
            Xs_train_class[int(Ys_train_set[i])].append(Xs_train_set[i,:])
            
            
        self.kde = []
        for i in range(len(Xs_train_class)):
            self.kde.append(KernelDensity(kernel='gaussian', bandwidth=self.bandwidth))
            self.kde[i].fit(Xs_train_class[i])
            
    def predict(self, samples): 
        classes = []
        scores = []
        for k in self.kde: 
            scores.append(k.score_samples(samples))
        for i in range(len(samples)): 
            l = []
            for j in range(len(scores)): 
                l.append(scores[j][i])
            classes.append(l.index(max(l)))
        return classes
    
    def score(self, Xs, Ys):
        classes = self.predict(Xs)
        return accuracy_score(Ys, classes)
        
    
    def set_params(self, **params): 
        self.bandwidth = params['bandwidth']
        
        

def calculate_confusion_matrix(classes):
    confusion_matrix = [0,# Real and Predicted Real
                        0,# Fake but Predicted Real
                        0,# Real but Predicted Fake
                        0]# Fake and Predicted Fake
    
    
    for i in range(len(classes)): 
        c = classes[i]
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
    print()
    print("Predictions \tReal\tFake")
    print("Real\t\t\t"+str(mat[0])+"\t\t"+str(mat[1]))
    print("Fake\t\t\t"+str(mat[2])+"\t\t"+str(mat[3]))
    

def crossValidation(Xs_r, Ys_r, paramName, triple, classifier):
    tmin = triple[0]
    tmax = triple[1]
    tstep = triple[2]

    validation_errs = []
    training_errs = []
    values = []
    
    

    folds = 5
    kf = StratifiedKFold(n_splits=folds)
    best_va_err = 100
    best_optimized = 0
    
    print("Starting Crossvalidation for " + paramName)

    #keep the +1 or it wont reach max
    for i in range(round((tmax - tmin)/tstep) + 1):
        
        #print(i, "of", round((tmax - tmin)/tstep))   
        values.append(round(tmin + i * tstep,3))
        classifier.set_params(**{paramName : values[i]})
        
        tr_err = va_err = 0
        for tr_ix,va_ix in kf.split(Ys_r,Ys_r):
            
            classifier.fit(Xs_r[tr_ix],Ys_r[tr_ix])
            
            tr_err += 1-classifier.score(Xs_r[tr_ix],Ys_r[tr_ix])
            va_err += 1-classifier.score(Xs_r[va_ix],Ys_r[va_ix])
        
        validation_errs.append(va_err/folds)
        training_errs.append(tr_err/folds)
        
          
        if(va_err < best_va_err):
              best_va_err = va_err
              best_optimized = round(tmin + i * tstep,3)
              
    print('best '+paramName+' value: ', best_optimized)
    print('best validation error: ', best_va_err/folds)
    
    
    plt.figure(figsize=(12, 8))
    plt.title("Crossvalidation for " + paramName)
    plt.axis([tmin,tmax,0,max(validation_errs)*1.15])
    plt.plot(values,training_errs,'-', label = "Training Error")
    plt.plot(values,validation_errs,'-', label = "Validation Error")
    plt.legend()
    plt.savefig(paramName+'.png', dpi=300)
    plt.show()
    plt.close()

    
    return best_optimized



def normalTest(mat):
    errors = mat[1] + mat[2]
    size = np.array(mat).sum()
    
    interval = size * (errors/size) *(1 - errors/size)
    interval = math.sqrt(interval) * 1.96
    
    return errors, interval

def mcnemarTest(classes1, classes2):

    #wrong in classifier 1 but right on 2    
    e01 = 0
    #right in classifier 1 but wrong on 2    
    e10 = 0
    for i in range(len(Ys_test)): 
        if Ys_test[i]==classes1[i] and Ys_test[i]==classes2[i]:
            #do nothing
            continue
            
        elif Ys_test[i]==classes1[i] and Ys_test[i]!=classes2[i]: 
            e10+=1
            
            
        elif Ys_test[i]!=classes1[i] and Ys_test[i]==classes2[i]:
            e01+=1
            
        else: 
            #do nothing
            continue
        
        
    top = abs(e01 - e10) - 1
    res = (top * top) / (e10 + e01)
    
    
        
    return res


"""
START OF PROGRAM

load, randomize and standardize
"""

# load data
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



"""
1.
Naïve Bayes classifier using Kernel Density Estimation
"""
print("Naive Bayes")

nb = OwnNaiveBaseClassifier(1)
bandwidth = crossValidation(Xs_train, Ys_train, 'bandwidth', (0.02,0.6,0.02), nb)

nb.set_params(**{'bandwidth': bandwidth})
nb.fit(Xs_train, Ys_train)
nb_classes = nb.predict(Xs_test)

nb_mat = calculate_confusion_matrix(nb_classes)

print_confusion_matrix(nb_mat)









"""
2.
Gaussian Naïve Bayes classifier

"""


print()
print("Guassian NB")

gnb = GaussianNB()
gnb.fit(Xs_train, Ys_train)
gnb_classes = gnb.predict(Xs_test)

gnb_mat = calculate_confusion_matrix(gnb_classes)

print_confusion_matrix(gnb_mat)




"""
3.Support Vector Machine with a Gaussian radial basis function
"""

print()
print("SVM")

sv = svm.SVC(C=1,kernel = 'rbf', gamma=0.2, probability=True)
gamma = crossValidation(Xs_train, Ys_train, 'gamma', (0.2, 6, 0.2), sv)

sv.set_params(**{'gamma': gamma})
sv.fit(Xs_train, Ys_train)
svm_classes = sv.predict(Xs_test)

svm_mat = calculate_confusion_matrix(svm_classes)
print_confusion_matrix(svm_mat)




"""
compare stuff
"""
print()
print("Normal tests")
error, interval = normalTest(nb_mat)
print(error, "+-", interval)

error, interval = normalTest(gnb_mat)
print(error, "+-", interval)

error, interval = normalTest(svm_mat)
print(error, "+-", interval)


print()
print("McNemar tests")

print("NB vs GNB:", mcnemarTest(nb_classes, gnb_classes))
print("NB vs SVM:", mcnemarTest(nb_classes, svm_classes))
print("GNB vs SVM:", mcnemarTest(gnb_classes, svm_classes))
print()

print("Bellow 3.84 means they might be equivalent but affected by random chance")












