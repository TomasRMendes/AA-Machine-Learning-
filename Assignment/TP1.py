import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.naive_bayes import  GaussianNB
from sklearn import svm



"""
aux
"""

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

def create_plot(X_r, Y_r, X_t, Y_t, feats, best_c):
    #create imege with plot for best classifier
    ax_lims=(-3,3,-3,3)
    plt.figure(figsize=(8,8))
    plt.title(str(feats)+ " features / C: " + str(best_c))
    plt.axis(ax_lims)
    reg = LogisticRegression(C=best_c, tol=1e-10)
    reg.fit(X_r[:,:feats],Y_r)
    
    
    # calc errors
    prob = reg.predict_proba(X_r[:,:feats])[:,1]
    squares = (prob-Y_r)**2
    training_err = round(np.mean(squares[:]),5)
    prob = reg.predict_proba(X_t[:,:feats])[:,1]
    squares = (prob-Y_t)**2
    test_err = round(np.mean(squares[:]),5)
    plt.text(1.5, 2.5, "Train: " + str(training_err)+ "\nTest: " + str(test_err))
    
    
    plotX,plotY,Z = poly_mat(reg,X_r,feats,ax_lims)
    plt.contourf(plotX,plotY,Z,[-1e16,0,1e16], colors = ('b', 'r'),alpha=0.5)
    plt.contour(plotX,plotY,Z,[0], colors = ('k'))
    plt.plot(X_r[Y_r>0,0],X_r[Y_r>0,1],'or')
    plt.plot(X_r[Y_r<=0,0],X_r[Y_r<=0,1],'ob')
    plt.plot(X_t[Y_t>0,0],X_t[Y_t>0,1],'xr',mew=2)
    plt.plot(X_t[Y_t<=0,0],X_t[Y_t<=0,1],'xb',mew=2)
    #plt.savefig('final_plot.png', dpi=300)
    plt.show()
    plt.close()
    # x is Test / o is Train


"""
standardize(train)
standardize(test based on train)

randomize(test)
randomize(train)

"""



mat_test = np.loadtxt('TP1_test.tsv',delimiter='\t')
mat_train = np.loadtxt('TP1_train.tsv',delimiter='\t')


# Ys contains the Classes (0 or 1)
Ys_test = mat_test[:,-1]
Ys_train = mat_train[:,-1]

# Xs contains the coordinates (x1, x2)
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
1.Naïve Bayes classifier using Kernel Density Estimation
"""



"""
#2.Gaussian Naïve Bayes classifier
"""

gnb = GaussianNB()
gnb.fit(Xs_train,Ys_train)

    






"""
3.Support Vector Machine with a Gaussian radial basis function
"""


def H_poly(X,Y,d):
 H = np.zeros((X.shape[0],X.shape[0]))
 for row in range(X.shape[0]):
     for col in range(X.shape[0]):
         k = (np.dot(X[row,:],X[col,:])+1)**d
         H[row,col] = k*Y[row]*Y[col]
 return H


svc_Xs_train=H_poly(Xs_train,Ys_train, 16)
svc_Xs_test=H_poly(Xs_test, Ys_test, 16)



def calc_fold(feats, X,Y, train_ix, valid_ix, C, gamma):
 sv = svm.SVC(C=C,kernel = 'rbf', gamma=gamma, probability=True)
 sv.fit(X[train_ix,:feats],Y[train_ix])
 prob = sv.predict_proba(X[:,:feats])[:,1]
 squares = (prob-Y)**2
 return np.mean(squares[train_ix]),np.mean(squares[valid_ix])



folds = 10
kf = StratifiedKFold(n_splits=folds)
C = 1
gamma = 1
best_va_err = 100000
best_feats = 0
for feats in range(2,16):
 tr_err = va_err = 0
 for tr_ix,va_ix in kf.split(Ys_train,Ys_train):
  r,v = calc_fold(feats,svc_Xs_train,Ys_train,tr_ix,va_ix, C, gamma)
  tr_err += r
  va_err += v
  
 if(va_err < best_va_err):
      best_va_err = va_err
      best_feats = feats
 print(feats,':', tr_err/folds,va_err/folds)

print('best: ',best_feats, best_va_err/folds)


#needs fixing
#create_plot(Xs_train, Ys_train, Xs_test, Ys_test, best_feats, 1)











