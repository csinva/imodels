from sklearn.cross_validation import train_test_split
from sklearn.datasets.mldata import fetch_mldata
from RuleListClassifier import *
from sklearn.ensemble import RandomForestClassifier

feature_labels = ["#Pregnant","Glucose concentration test","Blood pressure(mmHg)","Triceps skin fold thickness(mm)","2-Hour serum insulin (mu U/ml)","Body mass index","Diabetes pedigree function","Age (years)"]
    
data = fetch_mldata("diabetes") # get dataset
y = -(data.target-1)/2 # target labels (0: healthy, or 1: diabetes) - the original dataset contains -1 for diabetes and +1 for healthy

###############################################################################

Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y) # split

# train classifier (allow more iterations for better accuracy)
clf = RuleListClassifier(max_iter=10000, class1label="diabetes", verbose=False)
clf.fit(Xtrain, ytrain, feature_labels=feature_labels)

print "RuleListClassifier Accuracy:", clf.score(Xtest, ytest), "Learned interpretable model:\n", clf

###############################################################################

print "RandomForestClassifier Accuracy:", RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest)