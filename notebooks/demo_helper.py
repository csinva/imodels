import numpy as np
from scipy.io.arff import loadarff
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

path_to_current_file = os.path.dirname(os.path.abspath(__file__))


def viz_classification_preds(probs, y_test):
    '''look at prediction breakdown
    '''
    plt.subplot(121)
    plt.hist(probs[:, 1][y_test == 0], label='Class 0')
    plt.hist(probs[:, 1][y_test == 1], label='Class 1', alpha=0.8)
    plt.ylabel('Count')
    plt.xlabel('Predicted probability of class 1')
    plt.legend()

    plt.subplot(122)
    preds = np.argmax(probs, axis=1)
    plt.title('ROC curve')
    fpr, tpr, thresholds = metrics.roc_curve(y_test, preds)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot(fpr, tpr)
    plt.tight_layout()
    plt.show()


def get_ames_data():
    try:
        housing = fetch_openml(name="house_prices", as_frame=True, parser='auto')
    except:
        housing = fetch_openml(name="house_prices", as_frame=True)

    housing_target = housing['target'].values
    housing_data_numeric = housing['data'].select_dtypes('number').drop(columns=['Id']).dropna(axis=1)
    feature_names = housing_data_numeric.columns.values
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        housing_data_numeric.values, housing_target, test_size=0.75)
    return X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_names


def get_diabetes_data():
    '''load (classification) data on diabetes
    '''
    data = loadarff(os.path.join(path_to_current_file, "../tests/test_data/diabetes.arff"))
    data_np = np.array(list(map(lambda x: np.array(list(x)), data[0])))
    X = data_np[:, :-1].astype('float32')
    y_text = data_np[:, -1].astype('str')
    y = (y_text == 'tested_positive').astype(int)  # labels 0-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75)  # split
    feature_names = ["#Pregnant", "Glucose concentration test", "Blood pressure(mmHg)",
                     "Triceps skin fold thickness(mm)",
                     "2-Hour serum insulin (mu U/ml)", "Body mass index", "Diabetes pedigree function", "Age (years)"]
    return X_train, X_test, y_train, y_test, feature_names
