import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np
from itertools import izip

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix     
    N = X.shape[0]
    d = X.shape[1]
    uniqueClasses = np.unique(y)
    noUniqueClasses = len(uniqueClasses)
    y=np.squeeze(y,axis=1)
    means = np.array(np.mean(X[y==uniqueClasses[0]],axis=0))
    for i in xrange(1,noUniqueClasses):
        means=np.vstack((means,np.mean(X[y==uniqueClasses[i]],axis=0)))
    covmat = np.cov(X,rowvar=0)
    return means.T,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    N = X.shape[0]
    d = X.shape[1]
    uniqueClasses = np.unique(y)
    noUniqueClasses = len(uniqueClasses)
    y=np.squeeze(y,axis=1)
    means = np.array(np.mean(X[y==uniqueClasses[0]],axis=0))
    covmats = np.array([np.cov(X[y==uniqueClasses[0]],rowvar=0)])
    for i in xrange(1,noUniqueClasses):
        means = np.vstack((means,np.mean(X[y==uniqueClasses[i]],axis=0)))
        covmats = np.vstack((covmats,[np.cov(X[y==uniqueClasses[i]],rowvar=0)]))
    return means.T,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    invsigma = np.linalg.inv(covmat)
    detsigma = np.linalg.det(covmat)
    d = len(covmat)
    predictedLabels=[]
    for singleTest in Xtest:
        maximum = float('-inf')
        index=0
        for classNo,mean in enumerate(means.T):
            xMinusMean = singleTest - mean
            xMinusMean = xMinusMean.reshape(2,1)
            denominator = ((2*np.pi)**(d/2))*(detsigma**0.5)
            epiPower = -0.5*np.dot(np.dot(xMinusMean.T,invsigma),xMinusMean)
            value = np.exp(epiPower)/denominator
            if value > maximum:
                maximum = value
                index = classNo
        predictedLabels.append(float(index)+1)
    predictedLabels = np.array(predictedLabels)
    ytest=ytest.reshape(ytest.size)
    acc = 100*np.mean(predictedLabels == ytest)    
    return acc,predictedLabels

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    d = len(covmat)
    predictedLabels=[]
    for singleTest in Xtest:
        maximum = float('-inf')
        index=0
        for classNo,mean in enumerate(means.T):
            xMinusMean = singleTest - mean
            xMinusMean = xMinusMean.reshape(2,1)
            invsigma = np.linalg.inv(covmats[classNo])
            detsigma = np.linalg.det(covmats[classNo])
            denominator = ((2*np.pi)**(d/2))*(detsigma**0.5)
            epiPower = -0.5*np.dot(np.dot(xMinusMean.T,invsigma),xMinusMean)
            value = np.exp(epiPower)/denominator
            if value > maximum:
                maximum = value
                index = classNo
        predictedLabels.append(float(index)+1)
    predictedLabels = np.array(predictedLabels)
    ytest=ytest.reshape(ytest.size)
    acc = 100*np.mean(predictedLabels == ytest)    
    return acc,predictedLabels

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD
    # (X.T x X)-1 * X.T y                                                   
    return np.dot(np.linalg.inv(np.dot(X.T, X)) , np.dot(X.T, y))

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    #(lambda*identity + X.T * X)-1 *X.T * y
   # print X.shape
    #print y.shape
    XProduct = np.dot(X.T, X)
    identity = np.identity(XProduct.shape[0])
    lamb = np.multiply(identity, lambd)
    first = np.add( lamb, XProduct)
    second = np.dot(X.T, y)
    w = np.dot(np.linalg.inv(first), second)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    squaredDifference = np.square( np.subtract( ytest, np.dot(Xtest, w) ) )
    rmse = np.sum(squaredDifference)
    rmse /= Xtest.shape[0]
    return sqrt(rmse)

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    # error = 1/2(y-xw)'(y-xw) + 1/2 lambda * w'w
    # error_grad = -y'x + 1/2 w'x'x + 1/2 lamda*w'
    
    w = np.reshape(w,(w.size,1))
    # print "X"+str(X.shape)
    # print "w"+str(w.shape)
    yMinusXW = y - np.dot(X,w)
    # print "yMinusXW"+str(yMinusXW.shape)
    error = (0.5*np.dot((yMinusXW).T , yMinusXW)) + (0.5*lambd*np.dot(w.T,w))
    error = error.flatten()
    
    #-------------------------------------------Calculation of error Grad---------------------------------------------
    
    error_grad = (-1*np.dot(y.T,X))+(np.dot(np.dot(w.T,X.T),X))+(lambd*w.T)
    error_grad = np.reshape(error_grad, ((error_grad.size),1))
    error_grad = error_grad.flatten()
    """ 
    first = np.dot( np.dot(X.T, X), w)
    second =  np.dot(X.T, y)
    third = np.multiply(w, lambd)
    error_grad = first - second + third
    error_grad = error_grad.flatten()
    """
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    #Conver x to an array of scalars
    Xd = np.empty([x.shape[0], p+1])
    for i in range(x.shape[0]):
        for j in range(p+1):
            Xd[i][j] = x[i]**j      
    return Xd

# Main script

# Problem 1
# load the sample data 
                                                                
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# # QDA
means,covmats = qdaLearn(X,y)

qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
#plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
#plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

# plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
#plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
#plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
#mle_learn_data = testOLERegression(w, X, y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
#mle_learn_data_intercept = testOLERegression(w_i, X_i, y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

#print ('RMSE on training data without intercept')+str(mle_learn_data)
print('RMSE without intercept '+str(mle))
#print ('RMSE on training data with intercept ')+str(mle_learn_data_intercept)
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses1 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses1[i] = testOLERegression(w_l, X_i, y)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)
plt.plot(lambdas, rmses1)
plt.ylabel('RMSE')
plt.xlabel('Lambda')
plt.legend(('Test data','Training data'))
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
#plt.plot(lambdas,rmses4)
#plt.legend(('RMSE','Lambda')) 
#plt.show()

# # Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
#plt.plot(range(pmax),rmses5)
#plt.legend(('No Regularization','Regularization')) 
#plt.show()
