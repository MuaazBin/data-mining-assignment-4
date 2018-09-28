from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import graphviz


print('\n\n//=================== HOMEWORK 4 ====================//')
print('//----------------Developing a Classifier----------------//')
numSplits = 5
irisDataframe = pd.read_csv('../input/data.csv')


recordsPerSplit = len(irisDataframe) / numSplits
classifier = tree.DecisionTreeClassifier()

classes = set(irisDataframe['class'])

cleanValues = {
    'class': {

    }
}

for i in list(range(0, numSplits)):
    sliceIndexStart = i * recordsPerSplit
    sliceIndexEnd = sliceIndexStart + recordsPerSplit - 1
    irisTestingDataset = irisDataframe.loc[sliceIndexStart:sliceIndexEnd]
    irisTrainingDataset = irisDataframe.drop(irisTestingDataset.index[:])

    y = irisTrainingDataset['class']
    x = y.drop(columns='class')

    # classifier.fit(x, y)

    # dotData = tree.export_graphviz(classifier, out_file=None)
    # graph = graphviz.Source(dotData)

    # graph.render(f"../output/tree_{i}")



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
