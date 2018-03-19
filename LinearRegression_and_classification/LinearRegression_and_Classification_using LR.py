#Assignment:Task1 Regression Problem using house data, Task2 Classification problem using MNIST( hand-written digit data)
#Amruta Mangaonkar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


#Task1
#read train data from input files
traindata = pd.read_csv("housing_training.csv", sep=",", header=None)

X = np.matrix(traindata.iloc[:,:13])
new_col = np.ones((300,1))
X = np.hstack((new_col,X))
Y = np.matrix(traindata.iloc[:,13:])

#calculate optimal coefficients using linear regression
b = np.dot((np.dot(X.getT(),X)).getI(),np.dot(X.getT(),Y))
print(b)

#read test data from file
testdata = pd.read_csv("housing_test.csv", sep=",", header=None)

X_test = np.matrix(testdata.iloc[:,:13])
new_col = np.ones((206,1))
X_test = np.hstack((new_col,X_test))

Y_test = np.matrix(testdata.iloc[:,13:])

#Predict the values using test data and optimal coefficients
prediction = np.dot(X_test,b)
print(prediction)

#plot a graph
plt.plot(prediction,Y_test, 'ro')
plt.plot([0,60],[0,60])
plt.xlim(0,60)
plt.ylim(0,60)
plt.xlabel("Prediction")
plt.ylabel("Ground Truth")
plt.title("Prediction vs Ground Truth")
plt.show()

#calculate RMSE value
RMSE = ((np.sum(np.square(Y_test - prediction)))/X_test.shape[0])**(1/2)
print("RMSE:",RMSE)

#Task2

#read train data and labels from files
traindata = pd.read_csv("MNIST_15_15.csv", sep=",", header=None)
Y = pd.read_csv("MNIST_LABEL.csv", sep=",", header=None)

#convert label 5 and 6 into 1 and -1 respectively
Y[Y==5] = 1
Y[Y==6] = -1

#Normalize data
for indx in range(traindata.shape[1]):
    mean = np.mean(traindata.iloc[:,indx])
    std_dev = np.std(traindata.iloc[:,indx])
    if (std_dev!=0):
        traindata.iloc[:,indx] = (traindata.iloc[:, indx] - mean) / (std_dev)

FPR = []
TPR = []

#define set of lambdas
lambda_arr = [float('-inf'),-1.5,-1,-0.5,0,0.5,1,1.5,float('inf')]

#Include label with train data and shuffled data once
traindata = np.hstack((np.matrix(traindata),Y))
np.random.shuffle(traindata)

for cur_lambda in lambda_arr:
    FPR_sum =0
    TPR_sum =0
    print(cur_lambda)
    #used 10 fold cross validation
    kf = KFold(n_splits=10)
    cnt = 1
    for train_indx, test_indx in kf.split(traindata):
        print(cnt)
        cnt = cnt + 1
        #select train and test data from dataset
        train_data = traindata[train_indx]
        test_data = traindata[test_indx]

        new_col = np.ones((train_data.shape[0], 1))
        X = np.matrix(np.hstack((new_col,train_data[:,:225])))
        Y = train_data[:,225:]
        lambda_val = 0.1
        identity_m = np.identity(train_data.shape[1])

        #calculated optimal coefficients using ridge regression
        b = np.dot((((np.dot(X.getT(),X) + np.matrix(lambda_val * identity_m))).getI()),np.dot(X.getT(),Y))

        new_col = np.ones((test_data.shape[0],1))
        X_test = np.matrix(np.hstack((new_col,test_data[:,:225])))
        Y_actual = np.matrix(test_data[:,225:])
        #Calculate prediction using optimal coefficients and test data
        Y_prediction = np.dot(X_test,b)

        #classify prediction data using lambda
        result = Y_prediction > cur_lambda
        actual = Y_actual == 1

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        #calculate confusion matrix
        for indx in range(result.shape[0]):
            if(result[indx]== True and actual[indx]== True):
                TP = TP + 1
            elif (result[indx]==True and actual[indx]== False):
                FP = FP + 1
            elif(result[indx]== False and actual[indx]== True):
                FN = FN + 1
            else:
                TN = TN + 1

        TPR_temp = 0
        TPR_temp = TP / ( TP + FN )
        TPR_sum = TPR_sum + TPR_temp
        print(TPR_temp)

        FPR_temp = 0
        FPR_temp = FP / ( FP + TN )
        FPR_sum = FPR_sum + FPR_temp
        print(FPR_temp)

    #calculate average TPR and FPR values
    TPR.append(TPR_sum/10)
    FPR.append(FPR_sum/10)

#plot ROC curve
plt.plot(FPR,TPR)
plt.xlabel("TPR")
plt.ylabel("FPR")
plt.title("ROC Curve")
plt.show()

