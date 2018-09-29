from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


print('\n\n//=================== HOMEWORK 4 ====================//')
numSplits = 5
irisDataframe = pd.read_csv('../input/data.csv')


recordsPerSplit = len(irisDataframe) / numSplits
decisionTreeClassifier = tree.DecisionTreeClassifier()

classes = set(irisDataframe['class'])

print('\n//-------------------Cleaning Data-------------------//')
cleanValues = {
    'class': {}
}

print('Numeric Values of Categorical Data:')
i = 0
for value in classes:
    print(f'{value}: {i}')
    cleanValues['class'][value] = i
    i += 1

irisDataframe.replace(cleanValues, inplace=True)

print('\n//-------------------Constructing Models-------------------//')
decisionTreePerformance = {
    'accuracy': {},
    'fMeasure': {}
}

knnPerformance = {}

for i in list(range(0, numSplits)):
    # divide into training and testing data
    sliceIndexStart = i * recordsPerSplit
    sliceIndexEnd = sliceIndexStart + recordsPerSplit - 1
    irisTestingDataset = irisDataframe.loc[sliceIndexStart:sliceIndexEnd]
    irisTrainingDataset = irisDataframe.drop(irisTestingDataset.index[:])

    # divide into records and labels
    xTrainingSet = irisTrainingDataset.drop(columns='class')
    yTrainingSet = irisTrainingDataset['class']
    xTestingSet = irisTestingDataset.drop(columns='class')
    yTestingSet = irisTestingDataset['class']

    # make decision tree, calculate accuracy of predictions
    decisionTreeClassifier.fit(xTrainingSet, yTrainingSet)
    decisionTreePredictedLabels = decisionTreeClassifier.predict(xTestingSet)
    decisionTreeAccuracyScore = accuracy_score(yTestingSet, decisionTreePredictedLabels)
    decisionTreePerformance['accuracy'][i] = decisionTreeAccuracyScore

    # output decision tree
    dotData = tree.export_graphviz(decisionTreeClassifier, out_file=None)
    graph = graphviz.Source(dotData)
    graph.render(f"../output/tree_{i}")

    #  make knn classifier, calculate accuracy of predictions
    numNeighbors = list(range(1, 11))
    knnPerformance[i] = list()

    for k in numNeighbors:
        knnClassifier = KNeighborsClassifier(n_neighbors=k)
        knnClassifier.fit(xTrainingSet, yTrainingSet)
        knnPredictedValues = knnClassifier.predict(xTestingSet)
        knnAccuracyScore = accuracy_score(yTestingSet, knnPredictedValues)
        knnFScore = f1_score(yTestingSet, knnPredictedValues, average='weighted')

        knnPerformance[i].append({
            'k': k,
            'accuracy': knnAccuracyScore,
            'fScore': knnFScore
        })

# numFolds = 5
# decisionTreeClassifier = DecisionTreeClassifier()
# crossValidationScores = cross_val_score(decisionTreeClassifier, iris.data, iris.target, cv=numFolds)
#
# knnEntries = []
#
# for k in list(range(1,11)):
#     knnClassifier = KNeighborsClassifier(n_neighbors=k)
#     crossValidationScores = list(cross_val_score(knnClassifier, iris.data, iris.target, cv=numFolds))
#     newEntry = {
#         'k': k,
#         'cvScores': crossValidationScores
#     }
#     knnEntries.append(newEntry)
#
# print(knnEntries)
