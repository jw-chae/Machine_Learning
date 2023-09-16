import numpy as np
from numpy import random
import math

from pandas import DataFrame
#Likelihood 가능도(우도) PDF에서의 y값을 가능도로 본다. 정규분포를 따르는 PDF를 생각해보자. 0부터 100까지의 수직선에 있는 변수 중 50을 뽑을 확률은 원래는 0이지만 Likelihood에서는 0.4이다.
#PDF Probability Density Function 확률 밀도함수
#MLE Maximum Liklihood Estimator 최대 가능도 추정
#STD Standard Deviation 표준편차 (분산을 제곱근 한 것)
#Defining the class
class train_test_split:
    def __init__(self,data,target,test_size,datatype,shuffle=False):
        self._data = data
        self._target = target
        self._test_size = test_size
        self.shuffle = shuffle
        self.datatype = datatype
        #   self._xtrain = x_train
        #   self._xtest = x_test
        #   self._ytrain = y_train
        #   self._ytest = y_test
    @classmethod
    def split(self, data ,target ,test_size,shuffle=False,datatype='df'):
        #data.reset_index(drop=self.shuffle, inplace=self.shuffle) #데이터를 일단 섞어주기
        if(shuffle==True):
            if type(data) is np.ndarray:
                s = np.arange(data.shape[0])
                np.random.shuffle(s)
                data=data[s]
                target=target[s]
                print('data shuffle complete')       
            else:#(type(data) is DataFrame and shuffle is True):
                s = np.arange(data.value_counts.shape[0])
                np.random.shuffle(s)
                data=data[s]
                target=target[s]
                print('data shuffle complete')        


        if(datatype== 'df'):
            x_train = data.iloc[:round(len(data)*(1-test_size)),:]#0~0.7
            y_train = target.iloc[:round(len(target)*(1-test_size)),]
            x_test = data.iloc[round(len(data)*(1-test_size)):,:]#0.7~1
            y_test = target.iloc[round(len(target)*(1-test_size)):,]#0.7~1
        else:
            x_train = data[:round(len(data)*(1-test_size)),:]#0~0.7
            y_train = target[:round(len(target)*(1-test_size)),]
            x_test = data[round(len(data)*(1-test_size)):,:]#0.7~1
            y_test = target[round(len(target)*(1-test_size)):,]#0.7~1
        # x_train = x_train.to_numpy()
        # y_train = y_train.to_numpy()
        # x_test = x_test.to_numpy()
        # y_test = y_test.to_numpy()
        print("x_train: {} , x_test: {} ,y_train: {},y_test: {} ".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

        return x_train , x_test , y_train , y_test 


class accuracy:
    def __init__(self,y_test,y_pred):
        self.y_pred = y_test
        self.y_data = y_pred
    @classmethod
    def score(self,y_test,y_pred):
        acc=np.mean(y_test==y_pred)
        return acc

class MLE:
    def __init__(self, samples, m, std, learning_rate, epochs, verbose=False):
        """
        samples: MLE를 얻기위한 표본
        m: mean 평균
        std: 표준편차
        learning_rate: 가중치(weights) 업데이트를 위한 alpha 값
        epochs: training epochs
        verbose: status 출력을 할것인지 여부
        """
        self._samples = samples
        self._m = m
        self._std = std
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._verbose = verbose


    def likelihood(self, x, M):
        """
        확률 밀도함수 PDF는 정규분포 Normal distribution 입니다.
        PDF의 y 값은 likelihood와 같습니다.
        Likelihood = L(확률분포|관측값)
        x: 
        :return: likelihood of input x (likelihood of input x is same as y of pdf)
        """
       
        result = (1/np.sqrt(2*math.pi)*np.power(self._std, 2))*np.exp(-np.power(x-M,2)/(2*np.power(self._std, 2)))
    
        return np.prod(result)

    def fit(self):
        """
        훈련 평가
        우도를 최소화 하는 M은 경사 하강법으로 구할 수 있습니다.
        M은 샘플에 대한 MLE를 뜻합니다.
        """

        # M을 초기화 합니다.
        self._estimator = np.random.normal(self._m, self._std, 1)

        # epochs 횟수만큼 훈련을 진행합니다.
        self._training_process = []
        for epoch in range(self._epochs):
            likelihood = np.prod(self.likelihood(self._samples, self._m))
            prediction = np.prod(self.likelihood(self._samples, self._estimator))
            cost = self.cost(likelihood, prediction)
            self._training_process.append((epoch, cost))
            self.update(self._samples, self._estimator)

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))


    def cost(self, likelihood, prediction):
        """
        비용함수
        likelihood: 모집단의 우도
        :param prediction: 표본의 우도
        return: 최적화된 파라미터 값을 반환합니다.
        """
        return math.sqrt(likelihood - prediction)


    def update(self, x, M):
        """
        경사 하강법을 업데이트 합니다.
        경사는 추정치 입니다.
        x: 표본
        M: 추정치
        """
        gradient = np.sum(np.exp(-(np.power(x - M, 2) / (2*math.pow(self._std, 2)))))
        if self._m > self._estimator:
            self._estimator += self._learning_rate * gradient
        else:
            self._estimator -= self._learning_rate * gradient


    def get_mle(self):
        """
        parameter getter
        return: estimator of MLE
        """
        return self._estimator

class LinearRegression:
    def __init__(self, x , y):
        self.data = x
        self.label = y
        self.m = 0
        self.b = 0
        self.n = len(x)
    
    def compute_cost(self, x, y, theta):
 
        predictions = x.dot(theta)
        errors = np.subtract(predictions, y) 
        sqrErrors = np.square(errors) 
        J = 1 / (2 * self.m) * np.sum(sqrErrors)

        return J
        
    
    def gradient_descent(self, x, y, theta, alpha, iterations):
        cost_history = np.zeros(iterations) #반복하는 횟수만큼 cost를 저장할 0이 들어있는 행렬 만들기

        for i in range(iterations): #반복 시작
            predictions = x.dot(theta) #예측값은 X에 세타를 곱해준  벡터 내적
            errors = np.subtract(predictions, y) #에러는 예측값과 y를 빼준 값
            sum_delta = (alpha / self.m) * x.transpose().dot(errors)#세타의 수정값 sum_delta 공식 그대로 넣은것 
            theta = theta - sum_delta #수정된 세타의 값 

            cost_history[i] = self.compute_cost(x, y, theta) #cost를 계산한 값에 관한 기록을 cost history에 저장하자.   

        return theta, cost_history

   


class LogisticRegression:
    def __init__(self,x,y):      
        self.intercept = np.ones((x.shape[0], 1))  #y절편
        self.x = np.concatenate((self.intercept, x), axis=1)#x절편 concatenate는 배열 합치기
        self.weight = np.zeros(self.x.shape[1])#가중치
        self.y = y#y값 
         
    #Sigmoid method
    def sigmoid(self, x, weight):
        z = np.dot(x, weight)
        return 1 / (1 + np.exp(-z))
     
    #method to calculate the Loss 코스트 함수와 비슷한 개념이라 보며 된다.
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() 
     
    #Method for calculating the gradients
    def gradient_descent(self, X, h, y):
        return np.dot(X.T, (h - y)) / y.shape[0]#X.T Same as self.transpose()
 
     
    def fit(self, lr , iterations):
        for i in range(iterations):
            sigma = self.sigmoid(self.x, self.weight)
             
            loss = self.loss(sigma,self.y)
 
            dW = self.gradient_descent(self.x , sigma, self.y)
             
            #Updating the weights
            self.weight -= lr * dW
 
        return print('fitted successfully to data')
     
    #Method to predict the class label.
    def predict(self, x_new , treshold):
        x_new = np.concatenate((self.intercept, x_new), axis=1)
        result = self.sigmoid(x_new, self.weight)
        result = result >= treshold
        y_pred = np.zeros(result.shape[0])
        for i in range(len(y_pred)):
            if result[i] == True: 
                y_pred[i] = 1
            else:
                continue
                 
        return y_pred

#decision tree
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor 
        feature index : sepal and petal
            threshold : best split data of feature
                left  : left node
                right : node
                info_gain = information gain
        '''      
        # for decision node
        self.feature_index = feature_index 
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        # initialize the root of the tree 
        """ 
                root
            nodeL   nodeR    
        """
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        
        ''' 
        recursive function to build the tree
        X : attributes(sepal.lenth to petal.width)  
        Y : type (label)
        num_samples : row of feature
        num_features : column of feature
        ''' 
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X) #row and colum of features 
        #print(' sample:',num_samples, ' feature:',num_features," Class:",Y)
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            #print("best split:",best_split)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split 
                  1. take loop and list every feature of index
                  2. list every value of features
                  3. split dataset according to unique of feature values
                  4. dataset value could be null, so if dataset is not null compute impormation gain
                  5. information gain is as better as small, if current gain > max gain , change index
        '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        print('num_features:',num_features)
        # loop over all the features
        for feature_index in range(num_features): 
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data 
            if feature index is smaller than threshold(according to maximum gain), set in left array (node)
            if feature index is bigger than threshold, set in right array (node)
        '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            ratio = len(y[y == cls]) / len(y)
            entropy += -ratio * np.log2(ratio)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            ratio = len(y[y == cls]) / len(y)
            gini += ratio**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("Feature"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

# if __name__ == "__main__":

#     # samples for MLE
#     samples = np.array([64, 64.5, 65, 65.5, 66])

#     # assumptions about the population
#     mean = np.array([65.0])
#     std = 5

#     # get MLE
#     estimator = MLE(samples, mean, std, learning_rate=0.1, epochs=30, verbose=True)
#     estimator.fit()
#     result = estimator.get_mle()
#     print(result)