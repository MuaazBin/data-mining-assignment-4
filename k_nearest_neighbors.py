# ------------------------- Homework 4: Using and Evaluating a K Nearest Neighbors Model -------------------------- #
# Use k nearest neighbors model to classify iris dataset. Evaluate model's performance according to following
# metrics:
#   - Accuracy
#   - F1 Score
#
# Full problem description can be found in problem_description.pdf.
# ----------------------------------------------------------------------------------------------------------------- #

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('============================== HOMEWORK 4 ==============================')
    num_splits = 5
    flower_dataset = pd.read_csv('input/data.csv')

    records_per_split = len(flower_dataset) / num_splits
    decision_tree_classifier = tree.DecisionTreeClassifier()

    possible_flowers = set(flower_dataset['class'])

    print('\n------------ CONVERTING CATEGORICAL DATA TO NUMERIC VALUES -------------')
    clean_values = {
        'class': {}
    }

    print('CATEGORICAL DATA: NUMERIC VALUE')
    for index, flower in enumerate(possible_flowers):
        clean_values['class'][flower] = index
        print(f'{flower}: {index}')

    flower_dataset.replace(clean_values, inplace=True)

    print('\n--------------------- MAKING, EVALUATING MODELS ------------------------')
    decision_tree_performance = {
        'accuracy': {},
    }

    knn_performance = {}

    for i in range(0, num_splits):
        # divide into training and testing data
        slice_index_start = i * records_per_split
        slide_index_end = slice_index_start + records_per_split - 1
        flower_testing_dataset = flower_dataset.loc[slice_index_start:slide_index_end]
        flower_training_dataset = flower_dataset.drop(flower_testing_dataset.index[:])

        # divide into records and labels
        training_data = flower_training_dataset.drop(columns='class')
        training_labels = flower_training_dataset['class']
        testing_data = flower_testing_dataset.drop(columns='class')
        testing_labels = flower_testing_dataset['class']

        # make decision tree, calculate accuracy of predictions
        decision_tree_classifier.fit(training_data, training_labels)
        decision_tree_predicted_labels = decision_tree_classifier.predict(testing_data)
        decision_tree_accuracy = accuracy_score(testing_labels, decision_tree_predicted_labels)
        decision_tree_performance['accuracy'][i] = decision_tree_accuracy

        # output visualization of decision tree
        output = tree.export_graphviz(decision_tree_classifier, out_file=None)
        graph = graphviz.Source(output)
        graph.render(f"output/tree_{i}")

        #  make knn classifier, calculate accuracy of predictions
        num_neighbors = range(1, 11)

        for k in num_neighbors:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(training_data, training_labels)
            knn_predicted_values = knn_classifier.predict(testing_data)
            knn_accuracy = accuracy_score(testing_labels, knn_predicted_values)
            knn_f1_score = f1_score(testing_labels, knn_predicted_values, average='weighted')

            if k not in knn_performance:
                knn_performance[k] = list()

            knn_performance[k].append({
                'accuracy': knn_accuracy,
                'fScore': knn_f1_score
            })

    print('\n--------------- CALCULATING AVERAGE ACCURACY AND F SCORE ---------------')
    decision_tree_accuracy_scores = list(decision_tree_performance['accuracy'].values())
    average_decision_tree_accuracy = np.mean(decision_tree_accuracy_scores)
    print('Average Decision Tree Accuracy: ', average_decision_tree_accuracy)
    print('Average KNN Accuracy: Please see visualization.')
    print('Average KNN F Score: Please see visualization.')

    knn_average_accuracy = {}
    knn_average_f_score = {}
    for k in knn_performance:
        accuracy_list = [knn['accuracy'] for knn in knn_performance[k]]
        f_score_list = [knn['fScore'] for knn in knn_performance[k]]
        knn_average_accuracy[k] = np.mean(accuracy_list)
        knn_average_f_score[k] = np.mean(f_score_list)

    print('\n-------------------------- VISUALIZATIONS ------------------------------')
    print('A popup window should appear shortly.')
    plt.bar(list(knn_average_accuracy.keys()), list(knn_average_accuracy.values()))
    plt.title('Average Accuracy of K Nearest Neighbors')
    plt.xlabel('# of Nearest Neighbors')
    plt.ylabel('Average Accuracy')
    plt.axis([0, 11, 0.8, 1])
    plt.show()
    print('To view the next visualization, please exit out of the first popup window.')


    plt.bar(list(knn_average_f_score.keys()), list(knn_average_f_score.values()))
    plt.title('F Scores of K Nearest Neighbors')
    plt.xlabel('# of Nearest Neighbors')
    plt.ylabel('F Score')
    plt.axis([0, 11, 0.8, 1])
    plt.show()