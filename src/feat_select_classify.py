import numpy as np
import sklearn.cross_validation as cv
import sklearn.preprocessing as pp
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier


def read_CSV(filepath):
    """This function uses a filepath provided as a string representing an absolute 
    path to the CSV file. It then opens the file, reads in the contents, closes the
    file, and then returns the contents as a numpy matrix. 
    """
    data = np.genfromtxt(filepath, names=["ctm1", "ctm2", "age", "LOS", "status"], delimiter=",", skip_header=1)
    data = np.column_stack((data[column_name].astype("float64") for column_name in data.dtype.names))
    return data
    
def train_test_sets(data):
    """This function takes a dataset as a numpy array and splits the dataset into
    the features and the outcomes. It then utilizes the train_test_split function
    to split the dataset into training and testing sets with 40% of the data used
    for testing. Using StandardScaler, it then transforms the feature data and returns
    the scaled feature datasets and the outcome datasets.
    """
    features = data[:, 0:4]
    outcome = data[:, 4]
    features_train, features_test, outcome_train, outcome_test = cv.train_test_split(features, outcome, test_size=0.33, random_state=0)
    scaler = pp.StandardScaler().fit(features_train)
    features_train_ss = scaler.transform(features_train)
    features_test_ss = scaler.transform(features_test)
    return features_train_ss, features_test_ss, outcome_train, outcome_test
    
def feature_selection(features_train, outcome_train):
    """This functions takes two parameters from the original dataset, features_train and
    outcome_train that were split in the train_test_sets function. Using these features,
    this function utilizes three sklearn functions to carry out feature selection in order
    to compare which features are the most important within a dataset. It then returns
    the RFE, KBest, and Chi2 feature selection results, ranking the features of the dataset.
    """
    min_max_scaler = pp.MinMaxScaler().fit(features_train)
    features_train_minmax = min_max_scaler.transform(features_train)
    selector = SelectKBest(k="all")
    selector.fit(features_train, outcome_train)
    model = SVC(kernel="linear")
    rfe = RFE(model)
    rfe.fit(features_train, outcome_train)
    ch2 = SelectKBest(chi2, k="all")
    ch2.fit(features_train_minmax, outcome_train)
    return "RFE results: ", rfe.support_, rfe.ranking_, "KBest results: ", selector.scores_, "Chi2 results: ", ch2.scores_

def naive_bayes(features_train, features_test, outcome_train):
    """This function takes a dataset that was split into training and testing sets
    and scaled previously and then carries out the feature selection that was determined
    optimal from the feature_selection function. It then utilizes the GaussianNB function
    to carry out the classification prediction. It returns the outcome_predict and
    outcome_score from the classification to be further analyzed.
    """
    selector = SelectKBest(k=2)
    selector.fit(features_train, outcome_train)
    features_train_transform = selector.transform(features_train)
    features_test_transform = selector.transform(features_test)
    clf = GaussianNB()
    clf.fit(features_train_transform, outcome_train)
    outcome_predict = clf.predict(features_test_transform)
    outcome_score = clf.predict_proba(features_test_transform)[:,1]
    return outcome_predict, outcome_score

def logistic_regression(features_train, features_test, outcome_train):
    """This function takes a dataset that was split into training and testing sets
    and scaled previously and then carries out the feature selection that was determined
    optimal from a previous function. It then utilizes the LogisticRegression function
    to carry out the classification prediction. It returns the outcome_predict and
    outcome_score from the classification to be further analyzed.
    """
    selector = SelectKBest(k=2)
    selector.fit(features_train, outcome_train)
    features_train_transform = selector.transform(features_train)
    features_test_transform = selector.transform(features_test)
    clf = LogisticRegression()
    clf.fit(features_train_transform, outcome_train)
    outcome_predict = clf.predict(features_test_transform)
    outcome_score = clf.decision_function(features_test_transform)
    return outcome_predict, outcome_score
    
def SVC_GridSearch(features_train, features_test, outcome_train):
    """This function takes a dataset that was split into training and testing sets 
    and scaled previously and then utilizes GridSearchCV to determine the optimal
    kernel and parameters for the Support Vector Classifier on the dataset. The 
    function returns the overall best estimator, best parameters, and outcome predictions
    from the GridSearchCV output.
    """
    selector = SelectKBest(k=2)
    selector.fit(features_train, outcome_train)
    features_train_transform = selector.transform(features_train)
    features_test_transform = selector.transform(features_test)
    param_grid = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]},
    {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]},]
    clf = GridSearchCV(SVC(), param_grid, scoring='accuracy')
    clf.fit(features_train_transform, outcome_train)
    outcome_predict = clf.predict(features_test_transform)
    outcome_score = clf.decision_function(features_test_transform)
    return clf.best_estimator_, clf.best_params_, outcome_predict, outcome_score   

def classification_results(classifier, outcome_test, outcome_predict, outcome_score):
    """This functions takes the outcome_test, outcome_predict, and outcome_score parameters
    from a previously split dataset and classification predictor. It then carries
    out metrics on the classification predictor and returns the sensitivity, specificity,
    accuracy, f1 score, and auc.
    """
    cm = metrics.confusion_matrix(outcome_test, outcome_predict)
    tn = cm[0,0]
    neg = cm[0,1] + cm[0,0]
    sensitivity = metrics.recall_score(outcome_test, outcome_predict)
    specificity = tn/float(neg)
    accuracy = metrics.accuracy_score(outcome_test, outcome_predict)
    f1_score = metrics.f1_score(outcome_test, outcome_predict)
    auc = metrics.roc_auc_score(outcome_test, outcome_score)
    return ('Classifier: %s, Sensitivity: %.3f, Specificity: %.3f, Accuracy: %.3f, F1 score: %.3f, AUC: %.3f' 
    %(classifier, sensitivity, specificity, accuracy, f1_score, auc))
    
def random_forest(features_train, features_test, outcome_train):
    """This function takes a dataset that was split into training and testing sets
    and scaled previously.  It then utilizes the RandomForestClassifier function
    to carry out the classification prediction. It returns the outcome_predict and
    outcome_score, as well as the ranking of feature importances from the classification 
    to be further analyzed.
    """
    rfc = RandomForestClassifier(n_estimators=1000)
    rfc.fit(features_train, outcome_train)
    outcome_predict = rfc.predict(features_test)
    outcome_score = rfc.predict_proba(features_test)[:,1]
    return rfc.feature_importances_, outcome_predict, outcome_score

if __name__ == "__main__":    
    f_train, f_test, o_train, o_test = train_test_sets(read_CSV("data.csv"))
    NB_o_predict, NB_o_score = naive_bayes(f_train, f_test, o_train)
    LR_o_predict, LR_o_score = logistic_regression(f_train, f_test, o_train)
    best_estimator, best_params, svc_o_predict, svc_o_score = SVC_GridSearch(f_train, f_test, o_train)
    print classification_results("Gaussian Naive Bayes", o_test, NB_o_predict, NB_o_score)
    print classification_results("Logistic Regression", o_test, LR_o_predict, LR_o_score)
    print classification_results("SVC", o_test, svc_o_predict, svc_o_score)
    print best_estimator, best_params
    
    rfc_features, rfc_o_predict, rfc_o_score = random_forest(f_train, f_test, o_train)
    print classification_results("Random Forest", o_test, rfc_o_predict, rfc_o_score)
    print rfc_features
    print rfc_o_predict
    print rfc_o_score
