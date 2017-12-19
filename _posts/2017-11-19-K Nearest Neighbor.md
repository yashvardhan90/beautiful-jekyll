# K Nearest Neighbor

K Nearest Neighbor is a classification algorithm. It is used to classify data samples by measuring its distance with all other classes. The sample is assigned to the class by majority voting. The most common distance measure for KNN is Euclidean but other measures such as Hamming, Manhattan are also used based on the requirement and the dataset.

```python
import csv
import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import seaborn as sns
```


```python
def preprocessKNN(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        next(lines, None)
        dataset = list(lines)
        scaler = MinMaxScaler()
        scaler.fit(dataset)
        dataset1 = scaler.transform(dataset)
        for i in range(len(dataset1)):
            for j in range(len(dataset1[0])):
                dataset1[i][j] = float(dataset1[i][j])

        for row in dataset1:
            if row[-1] > 0:
                row[-1] =1;
            trainingSet.append(row)

            # if random.random() < split:
            #     trainingSet.append(row)
            # else:
            #         testSet.append(row)
```


```python
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(4,length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
```


```python
def hammingDistance(instance1, instance2):
    distance =0
    for i in range(4):
        if(instance1[i]!= instance2[i]):
            distance+=1
    return distance
```


```python
def KNNImpl(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        dist += hammingDistance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    votes = [0]*2
    for x in range(len(neighbors)):
        votes[int(neighbors[x][-1])] +=1
    return votes.index(max(votes))
```


```python
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0
```


```python
def KNNClassifier():
    trainingSet = []; testSet = []; dummy = []
    split = 0.8
    preprocessKNN('train.csv', split, trainingSet, dummy)
    preprocessKNN('test.csv',split,testSet,dummy)
    predictions = []
    k = 9
    for x in range(len(testSet)):
        result = KNNImpl(trainingSet, testSet[x], k)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('KNN Accuracy: ' + repr(accuracy) + '%')
    plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13],[45,49,43,50,45,49,47,50,39,45,39,45,39])
    plt.xlabel('K')
    plt.ylabel('Accuracy with hamming distance')
    plt.show()
```
