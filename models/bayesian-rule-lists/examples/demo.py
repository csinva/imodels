from RuleListClassifier import *
import sklearn.ensemble
from sklearn.cross_validation import train_test_split
from sklearn.datasets.mldata import fetch_mldata

dataseturls = ["https://archive.ics.uci.edu/ml/datasets/Iris", "https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes"]
datasets = ["iris", "diabetes"]
data_feature_labels = [
    ["Sepal length", "Sepal width", "Petal length", "Petal width"],
    ["#Pregnant","Glucose concentration demo","Blood pressure(mmHg)","Triceps skin fold thickness(mm)","2-Hour serum insulin (mu U/ml)","Body mass index","Diabetes pedigree function","Age (years)"]
]
data_class1_labels = ["Iris Versicolour", "No Diabetes"]
for i in range(len(datasets)):
    print "--------"
    print "DATASET: ", datasets[i], "(", dataseturls[i], ")"
    data = fetch_mldata(datasets[i])
    y = data.target
    y[y>1] = 0
    y[y<0] = 0

    Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y)    
    
    clf = RuleListClassifier(max_iter=50000, n_chains=3, class1label=data_class1_labels[i], verbose=False)
    clf.fit(Xtrain, ytrain, feature_labels=data_feature_labels[i])
    
    print "accuracy:", clf.score(Xtest, ytest)
    print "rules:\n", clf
    print "Random Forest accuracy:", sklearn.ensemble.RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest)