'''
learnmodel.py:
   Usage:
           `python learnmodel.py [model]`
   Where:
           `model` in ['AdaBoost', 'DecisionTree', 'GNB', 'GradientBoosting', \
                       'KNN', 'RandomForest', 'ALL']
           if 'NONE' is specified, all algorithms will be tested and the best one will be selected.

Wikipedia entry for each algorithm for basic overview:
 0. AdaBoost          / "Adaptive Boosting"
                        https://en.wikipedia.org/wiki/AdaBoost
 1. DecisionTree      / "Decision Tree Learning"
                        https://en.wikipedia.org/wiki/Decision_tree_learning
 2. GNB               / "Gaussian Naive-Bayes Classifier"
                        https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes
 3. GradientBoosting  / "Gradient Boosting"
                        https://en.wikipedia.org/wiki/Gradient_boosting
 4. KNN               / "K-nearest neighbors (classification)"
                        https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
 5. RandomForest      / "Random Forest Ensemble Learning"
                        https://en.wikipedia.org/wiki/Random_forest
    NONE              / Run each algorithm and choose the best one

scikit-learn documentation:
 http://scikit-learn.org/stable/modules/ensemble.html
'''

import sys
print '\033[1;32mLaunched %s successfully.\033[0m' % (sys.argv[0])
print 'Importing packages.'
import os
#SciPy Ecosystem:
import numpy, pandas, pickle, sklearn.ensemble as ske
from sklearn import cross_validation, tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
print '\033[1;32mImported packages successfully.\033[0m'

DATA_PATH = './data.csv'
CLASSIFIER_PATH = './classifier'
ALLOWED = ['AdaBoost', 'DecisionTree', 'GNB', 'GradientBoosting', 'KNN', 'RandomForest', 'ALL']
ALGORITHMS = {
    'AdaBoost': ske.AdaBoostClassifier(n_estimators=100),
    'DecisionTree': tree.DecisionTreeClassifier(max_depth=10),
    'GNB': GaussianNB(),
    'GradientBoosting': ske.GradientBoostingClassifier(n_estimators=50),
    'KNN': BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
    'RandomForest': ske.RandomForestClassifier(n_estimators=50),
}

class InputException(Exception):
    def handle(self):
        print self.message

def main():
    model = read_user_input()
    sanitize_input(model)
    X,Y,data = import_data(DATA_PATH)
    X_train, X_test, Y_train, Y_test, features = select_features(X,Y,data)
    test_algorithm(model, features, X_train, X_test, Y_train, Y_test)

def read_user_input():
    args = sys.argv
    if len(args) != 2:
        raise InputException(
            '\n\t\033[0;31mError on argparse, exactly one model should be specified.\033[0m'
            '\n\t\033[0;31mUsage: "learnmodel.py [model]"\033[0m'
            '\n\t\t\033[0;31mmodel: [AdaBoost, DecisionTree, GNB, GradientBoosting, KNN, RandomForest, ALL]\033[0m'
            '\n\t\t\033[0;31m"ALL" will test all models and return the most accurate.\033[0m'
        )
    return args[1]

def sanitize_input(model):
    '''
    Make sure that the inputs (data.csv and the user-specified model) are valid.
        input:
           - data.csv (from DATA_PATH)
           - model stipulated by `read_user_input`
           - "classifier" directory exists (errors if we try to write to a non-existent one)
        output: None on success, reports error to user if one occurs
    '''
    if not os.path.isfile(DATA_PATH):
        raise InputException(
            '\n\t\t\033[0;31mError on import, "data.csv" was not found.\033[0m'
        )
    if not os.path.isdir(CLASSIFIER_PATH):
        raise InputException(
            '\n\t\t\033[0;31mError on import, could not find the "classifier" directory.\033[0m'
            '\n\t\t\033[0;31m`mkdir classifier`, then try running again.\033[0m'
        )
    if (sys.argv[1] not in ALLOWED):
        print '\033[0;31mYou chose: %s\033[0m' % model
        raise InputException(
            '\n\t\033[0;31mError on argparse, model not supported.\033[0m'
            '\n\t\033[0;31mUsage: "learnmodel.py [model]"\033[0m'
            '\n\t\t\033[0;31mmodel: [AdaBoost, DecisionTree, GNB, GradientBoosting, KNN, RandomForest, ALL]\033[0m'
            '\n\t\t\033[0;31m"ALL" will test all models and return the most accurate.\033[0m'
        )

def import_data(file_to_read):
    print 'Importing features from %s' % file_to_read
    data = pandas.read_csv(file_to_read, sep='|')
    '''
    Drop features that we don't want to learn from:
       Name: an identifier for our benefit, if an attacker isn't inept they probably
             won't actually name something "maliciousvirus.exe"
       md5:  byte sums can be useful for traditional antivirus systems, since they can
             compare files on the user's computer against known malicious sums.
       legitimate: this is the human-labeled answer for validation. If a .exe actually
             had this value in the header antivirus would not be necessary. Learning a
             model that has this value is useless.
    '''
    #the values for training/testing:
    X = data.drop(['Name','md5','legitimate'], axis=1).values
    #the answers for testing: store the answers for each file
    Y = data['legitimate'].values
    return X,Y,data

def select_features(x,y,data):
    '''
    Use a tree classifier to select the most relevent features from data.csv
    70%-30% train-test split for purposes of cross validation.
    '''
    feature_select = ske.ExtraTreesClassifier().fit(x,y)
    model = SelectFromModel(feature_select, prefit=True)
    x_new = model.transform(x)
    nb_features = x_new.shape[1]
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_new, y, test_size=0.3)
    features = []
    print '%i features were selected as being important:' % nb_features
    indices = numpy.argsort(feature_select.feature_importances_)[::-1][:nb_features]
    col_width = len(max(data.columns[2+indices[f]] for f in range(nb_features))) + 5
    for f in range(nb_features):
        number = f+1
        feature_name = ''.join(data.columns[2+indices[f]].ljust(col_width))
        feature_importance = feature_select.feature_importances_[indices[f]]
        print '   %d.\t%s %f%%' % (number, feature_name, (feature_importance * 100))
    for f in sorted(numpy.argsort(feature_select.feature_importances_)[::-1][:nb_features]):
        features.append(data.columns[2+f])
    return x_train, x_test, y_train, y_test, features

def test_algorithm(algorithm, features, X_train, X_test, Y_train, Y_test):
    results = {}
    number = 1
    col_width = max(len(key) for key in ALGORITHMS) + 2
    if algorithm == 'ALL':
        print '\n%s specified: testing all available algorithms then selecting most accurate.' % algorithm
        for algo in ALGORITHMS:
            classifier = ALGORITHMS[algo]
            algo_print = ''.join(algo.ljust(col_width))
            classifier.fit(X_train, Y_train)
            score = classifier.score(X_test, Y_test)
            print '   %d.\t%s %f%%' % (number, algo_print, score*100)
            number += 1
            results[algo] = score
    elif algorithm in ALGORITHMS.keys():
        print '\nTesting the %s algorithm.' % algorithm
        algo = algorithm
        algo_print = ''.join(algo.ljust(col_width))
        classifier = ALGORITHMS[algorithm]
        classifier.fit(X_train, Y_train)
        score = classifier.score(X_test, Y_test)
        print '   %d.\t%s %f%%' % (number, algo_print, score*100)
        results[algo] = score
    else:
        raise InputException(
            '\n\t\033[0;31mError, I am not sure how you got into this branch.\033[0m'
            '\n\t\033[0;31m       This should not be able to happen.\033[0m'
        )
    winner = max(results, key=results.get)
    print('\nMost effective algorithm is %s with a %f %% success' % (winner, results[winner]*100))
    
    # Save the algorithm and the feature list for later predictions
    print('Saving algorithm and feature list in classifier directory...')
    joblib.dump(ALGORITHMS[winner], 'classifier/classifier.pkl')
    open('classifier/features.pkl', 'w').write(pickle.dumps(features))
    print('\033[1;32mSaved!\033[0m')
    
    # Identify false positive and false negative rates
    print '\nIdentifying false positive and false negative rates (takes longer for certain algorithms):'
    classifier = ALGORITHMS[winner]
    results = classifier.predict(X_test)
    mt = confusion_matrix(Y_test, results)
    print '   False positive rate : %f%%' % ((mt[0][1] / float(sum(mt[0])))*100)
    print '   False negative rate : %f%%' % ((mt[1][0] / float(sum(mt[1]))*100))

if __name__=='__main__':main()
