import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.naive_bayes import  GaussianNB
from sklearn import svm



"""
aux
"""

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
    
    def set_params(self, **params): 
        self.bandwidth = params['bandwidth']
        
        


def poly_mat(reg,X_data,feats,ax_lims):
    #create score matrix for contour
    Z = np.zeros((200,200))
    xs = np.linspace(ax_lims[0],ax_lims[1],200)
    ys = np.linspace(ax_lims[2],ax_lims[3],200)
    X,Y = np.meshgrid(xs,ys)
    points = np.zeros((200,2))
    points[:,0] = xs
    for ix in range(len(ys)):
        points[:,1] = ys[ix]
        x_points=H_poly(points,ys,16)[:,:feats]
        Z[ix,:] = reg.decision_function(x_points)
    return (X,Y,Z)

    

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
    print("Predictions \tReal\tFake")
    print("Real\t\t\t"+str(mat[0])+"\t\t"+str(mat[1]))
    print("Fake\t\t\t"+str(mat[2])+"\t\t"+str(mat[3]))
    



def H_poly(X,Y,d):
 H = np.zeros((X.shape[0],X.shape[0]))
 for row in range(X.shape[0]):
     for col in range(X.shape[0]):
         k = (np.dot(X[row,:],X[col,:])+1)**d
         H[row,col] = k*Y[row]*Y[col]
 return H



#implement the triple thing
#triple [min, max, step]
def crossValidation(Xs_r, Ys_r, paramName, triple, classifier):
    tmin = triple[0]
    tmax = triple[1]
    tstep = triple[2]

    params = dict()
    params[paramName] = 0


    folds = 5
    kf = StratifiedKFold(n_splits=folds)
    best_va_err = 100
    best_optimized = 0
    
    print("Starting Crossvalidation for " + paramName)

    #keep the +1 or it wont reach max
    for i in range(round((tmax - tmin)/tstep) + 1):
        
        #print(i, "of", round((tmax - tmin)/tstep))   
            
        params[paramName] = round(tmin + i * tstep,3)
        classifier.set_params(**params)
        
        tr_err = va_err = 0
        for tr_ix,va_ix in kf.split(Ys_r,Ys_r):
            
            classifier.fit(Xs_r[tr_ix],Ys_r[tr_ix])
            if paramName == 'gamma': 
                prob = np.round(classifier.predict_proba(Xs_r[:,:])[:,1])
            else: 
                prob = classifier.predict(Xs_r)    
            squares = (prob-Ys_r)**2
            r = np.mean(squares[tr_ix])
            v = np.mean(squares[va_ix])   
              
            tr_err += r
            va_err += v
          
        if(va_err < best_va_err):
              best_va_err = va_err
              best_optimized = round(tmin + i * tstep,3)
              
    print('best '+paramName+' value: ', best_optimized)
    print('best validation error: ', best_va_err/folds)
    
    return best_optimized
    


"""
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


nb = OwnNaiveBaseClassifier(1)
print("starting cross val")
bandwidth = crossValidation(Xs_train, Ys_train, 'bandwidth', (0.2,0.6,0.02), nb)
nb.set_params(**{'bandwidth': bandwidth})
nb.fit(Xs_train, Ys_train)
classes = nb.predict(Xs_test)
print_confusion_matrix(calculate_confusion_matrix(classes))



"""
2.
Gaussian Naïve Bayes classifier


gnb = GaussianNB()
gnb.fit(Xs_train,Ys_train)
"""



"""
3.Support Vector Machine with a Gaussian radial basis function
"""

sv = svm.SVC(C=1,kernel = 'rbf', gamma=0.2, probability=True)
gamma = crossValidation(Xs_train, Ys_train, 'gamma', (0.2, 6, 0.2), sv)
sv.set_params(**{'gamma': gamma})
sv.fit(Xs_train, Ys_train)
classes = np.round(sv.predict_proba(Xs_test)[:,1])
print_confusion_matrix(calculate_confusion_matrix(classes))








