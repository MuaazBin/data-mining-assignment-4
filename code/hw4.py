from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


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

    for k in numNeighbors:
        knnClassifier = KNeighborsClassifier(n_neighbors=k)
        knnClassifier.fit(xTrainingSet, yTrainingSet)
        knnPredictedValues = knnClassifier.predict(xTestingSet)
        knnAccuracyScore = accuracy_score(yTestingSet, knnPredictedValues)
        knnFScore = f1_score(yTestingSet, knnPredictedValues, average='weighted')

        if k not in knnPerformance:
            knnPerformance[k] = list()

        knnPerformance[k].append({
            'accuracy': knnAccuracyScore,
            'fScore': knnFScore
        })

decisionTreeAccuracyScores = list(decisionTreePerformance['accuracy'].values())
averageDecisionTreeAccuracy = np.mean(decisionTreeAccuracyScores)

kAccuracies = {}
for k in knnPerformance:
    accuracyValuesList = [knn['accuracy'] for knn in knnPerformance[k]]

    kAccuracies[k] = {
        'accuracyValues': accuracyValuesList,
        'averageAccuracy': np.mean(accuracyValuesList)
    }

plt.bar(list(kAccuracies.keys()), [knn['accuracyValues'] for knn in knnPerformance])
plt.show()