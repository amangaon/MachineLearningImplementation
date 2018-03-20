# Home Work 1
# Machine Learning
# Name : Amruta Mangaonkar
# KNN Implementation

import numpy as np
import pandas as pd
from collections import Counter

#read train and test data from input files
traindata = pd.read_csv("MNIST_training.csv")
testdata = pd.read_csv("MNIST_test.csv")

#convert data into features and label matrices
#Train features and labels
X = np.matrix(traindata.iloc[:,1:])
Y = np.matrix(traindata.iloc[:,:1])

#Test features and labels
X_test = np.matrix(testdata.iloc[:,1:])
Y_test = np.matrix(testdata.iloc[:,:1])

#List of different K-values to be considered
kElemValueList = [1,2,3,4,5,6,7,8,9]
for val in kElemValueList:
    kElemValue = val
    print("kElemValue: ",kElemValue)
    identicalCnt = 0
    for indx in range(X_test.shape[0]):
        #Test vector
        vX_test = X_test[indx,:]
        euclidean_dist = []
        kElem = []
        for indx1 in range(X.shape[0]):
            #Train vector
            vX_train = X[indx1,:]
            #Calculate euclidean distance
            d = np.sqrt(np.sum(np.square(vX_test - vX_train)))
            #Collect euclidean distance of every test sample with all train samples
            euclidean_dist.append((indx1,d))
        #Sort Euclidean distances in ascending order
        sortedDist=sorted(euclidean_dist, key=lambda tup: tup[1])
        #Select first k elements with shortest euclidean distance
        kElem = sortedDist[:kElemValue]
        kElemList = []
        for tup in kElem:
            kElemList.append(tup[0])
        #Create a list of labels of k selected elements
        kLabels = []
        for tup in kElemList:
            kLabels.append(int(Y[tup]))
        #Sort labels
        kLabels = sorted(kLabels)
        #Group the labels into 10 groups and take a count followed by sorting in reverse order
        kLabelCnt = []
        kLabelCnt = sorted(Counter(kLabels).items(), key=lambda tup: tup[1], reverse=True)
        #Select label that occured maximum time
        maxCntLabel = kLabelCnt[0]
        prediction = maxCntLabel[0]
        ground_truth = Y_test[indx]
        #Check if prediction is identical to ground truth
        if(prediction == ground_truth):
            identicalCnt = identicalCnt + 1

    #Calculate Accuracy
    accuracy = identicalCnt/Y_test.shape[0] * 100
    print(identicalCnt)
    print(accuracy)
