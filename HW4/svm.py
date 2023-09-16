import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.solvers

"""
apply Lagrange multiplier with kkt 
L(x,lamda)=F(x)+lamda*h(x)
update w*xi +b >1 for two cases
yi=1  and yi=-1
for the first data which make the boundary, we call it support vector. 
so yi equals to {-1,1}

if satisty the condition, apply update rule to update gradients
the rule is as so:
w= w-lr*dw =w-lr*2*lamda*w 
b = b-lr*db = b 
else:
    w=w-lr*dw =w-lr(*2*lamda*w-yi*xi)
    b = b-lr*db = b - lr*yi

"""
class Hard_graidnet_SVM:

    def __init__(self, learning_rate=0.001, alpha_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.alpha_param = alpha_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
   
    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        # init weights
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.alpha_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.alpha_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        pred_result = np.sign(approx)
        return pred_result



def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=1):
        self.kernel = kernel #K[i,j] =kernel(X[i], X[j])
        self.C = C


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Create values for the quadratic programming
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
        
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples),'d')
        b = cvxopt.matrix(0.0)

        #if C is none, apply hard margin else apply soft maergin
        #modify G, h so that we have a soft-margin classifier
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_std = np.diag(np.ones(n_samples) * -1)#make diagonal matrix
            G_slack = np.identity(n_samples)#make identity matrix size of n_samples
            G = cvxopt.matrix(np.vstack((G_std, G_slack)))
            h_std = np.zeros(n_samples)
            h_slack = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((h_std, h_slack)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        # solving a quadratic progaming 
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        #add slack variables in range 0 to 1
        min_multiplier = 1e-3

        sv = a > min_multiplier
        indices = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):

            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[indices[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
    
    
    def predict(self, X):
        if self.w is not None:
            pred =  np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            pred = y_predict + self.b
        return np.sign(pred)


class OneVsRestSVM:
    def __init__(self, n_classes=3):#get iris data
        self.n_classes = n_classes
        self.clfs = []
        self.y_pred = []
    
    # onehot encoding y variable 
    def one_vs_rest_labels(self, y_train):
        y_train = pd.get_dummies(y_train)
        return y_train
    
    # get encoded y and loop for the number of classes
    def fit(self, X_train, y_train, gamma=0.001):
        # y encoding
        y_encoded = self.one_vs_rest_labels(y_train)
        
        for i in range(self.n_classes):
            clf = Hard_graidnet_SVM()
            clf.fit(X_train, y_encoded.iloc[:,i])
            self.clfs.append(clf)

    # voting based on the results from each classifier
    def predict(self, X_test):
        vote = np.zeros((len(X_test), 3), dtype=int)
        size = X_test.shape[0]
        
        for i in range(size):
            #vote for class belonging the class +1
            #else give -1
            if self.clfs[0].predict(X_test)[i] == 1:
                vote[i][0] += 1
                vote[i][1] -= 1
                vote[i][2] -= 1
            elif self.clfs[1].predict(X_test)[i] == 1:
                vote[i][0] -= 1
                vote[i][1] += 1
                vote[i][2] -= 1
            elif self.clfs[2].predict(X_test)[i] == 1:
                vote[i][0] -= 1
                vote[i][1] -= 1
                vote[i][2] += 1
    
            # 투표한 값 중 가장 큰 값의 인덱스를 test label에 넣는다
            self.y_pred.append(np.argmax(vote[i]))

        # test를 진행하기 위해 0,1,2로 되어있던 데이터를 다시 문자 label로 변환
        self.y_pred = pd.DataFrame(self.y_pred).replace({0:'setosa', 1:'versicolor', 2:'virginica'})
        return self.y_pred

    def evaluate(self, y_test):
        print('Accuacy : {: .5f}'.format(accuracy_score(y_test, self.y_pred)))
def feature_normalize(X):

  #Note here we need mean of indivdual column here, hence axis = 0
  mu = np.mean(X, axis = 0)  
  # Notice the parameter ddof (Delta Degrees of Freedom)  value is 1
  sigma = np.std(X, axis= 0, ddof = 1)  # Standard deviation The reason for specifying the standard deviation ddof is that numpy's std default is 0
  X_norm = (X - mu)/sigma
  return X_norm #, mu, sigma
# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    df = pd.read_csv('D:\\OneDrive\\Documents\\SJTU\\기계학습(머신러닝)\\Machine_Learning\\iris.data')
    df.reset_index(drop=True, inplace=True)
    #df.info()
    m, n = df.shape
    print('Number of training examples m = ', m)
    print('Number of features n = ', n - 1)
    X= df.iloc[:100,:2] #data for training
    X = feature_normalize(X)
    y = df.iloc[:100,-1] #target
    X = X.to_numpy()
    y= y.to_numpy()
    y=np.where(y==0,-1,1)#ternary operator
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)#shuffle and take train test split

    # clf = SVM()
    # clf.fit(X_train, y_train)
    # predictions = clf.predict(X_test)
    onevsrest = OneVsRestSVM()
    onevsrest.fit(X_train, y_train)
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    #print("SVM classification accuracy", accuracy(y_test, predictions))

    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)   
        plt.xlabel('sepal length (cm)')
        plt.ylabel('petal lenth (cm)')
        plt.scatter(X[:, 0], X[:, 1], marker="o",c=y)
  
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)#get hyper plane crossing line
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)#get hyper plane minus values
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)#get hyber plane plus vaues
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    visualize_svm()