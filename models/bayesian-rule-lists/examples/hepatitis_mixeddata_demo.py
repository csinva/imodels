from sklearn.cross_validation import train_test_split
from sklearn.datasets.mldata import fetch_mldata
from RuleListClassifier import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

"""
https://archive.ics.uci.edu/ml/datasets/Hepatitis
1. Class: DIE, LIVE 
2. AGE: 10, 20, 30, 40, 50, 60, 70, 80 
3. SEX: male, female 
4. STEROID: no, yes 
5. ANTIVIRALS: no, yes 
6. FATIGUE: no, yes 
7. MALAISE: no, yes 
8. ANOREXIA: no, yes 
9. LIVER BIG: no, yes 
10. LIVER FIRM: no, yes 
11. SPLEEN PALPABLE: no, yes 
12. SPIDERS: no, yes 
13. ASCITES: no, yes 
14. VARICES: no, yes 
15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00 
-- see the note below 
16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250 
17. SGOT: 13, 100, 200, 300, 400, 500, 
18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0 
19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90 
20. HISTOLOGY: no, yes 
""" 
data = fetch_mldata("datasets-UCI hepatitis") # get dataset

#some data cleaning (due to horrible mldata format)
# target
y = [1 if 'live' in v[0].lower() else 0 for v in data['Class'][0]]
# categorical variables
data['SEX'] = data.data
feature_labels = [col for col in data['COL_NAMES'] if col == col.upper()]
columns = {}
for label in feature_labels:
    column = data[label] if len(data[label])>1 else data[label][0]
    while type(column[0]) == list or type(column[0]) == np.ndarray:
        column = [c[0] for c in column]
    columns[label] = pd.Series(column)
# numeric variables
columns['AGE'] = data.target 
columns['BILIRUBIN'] = pd.Series(data['double1'][0])
columns['ALK PHOSPHATE'] = pd.Series(data['int2'][0])
columns['ALK PHOSPHATE'][columns['ALK PHOSPHATE']<0] = np.nan
columns['SGOT'] = pd.Series(data['int2'][1])
columns['SGOT'][columns['SGOT']<0] = np.nan
columns['ALBUMIN'] = pd.Series(data['double3'][0])
columns['PROTIME'] = pd.Series(data['int4'][0])
columns['PROTIME'][columns['PROTIME']<0] = np.nan
# convert to dataframe
hepatitis_df = pd.DataFrame(columns)
# deal with missing values
for c in hepatitis_df.columns:
    if hepatitis_df[c].dtype != np.object:
        hepatitis_df[c] = hepatitis_df[c].fillna(hepatitis_df[c][~np.isnan(hepatitis_df[c])].mean())

print hepatitis_df.head()

###############################################################################

Xtrain, Xtest, ytrain, ytest = train_test_split(hepatitis_df, y) # split

# train classifier (allow more iterations for better accuracy)
clf = RuleListClassifier(max_iter=10000, class1label="survival", verbose=False)
clf.fit(Xtrain, ytrain)

print "RuleListClassifier Accuracy:", clf.score(Xtest, ytest), "Learned interpretable model:\n", clf

###############################################################################

try:
    from category_encoders import HashingEncoder
except:
    raise Exception("Please install category_encoders (pip install category_encoders) for comparing mixed data with Random Forests!")
from sklearn import pipeline

ppl = pipeline.Pipeline([
    ('encoder', HashingEncoder(cols=['LIVER_BIG', 'ANTIVIRALS', 'HISTOLOGY', 'SEX', 'STEROID', 'MALAISE', 'FATIGUE', 'SPIDERS', 'VARICES', 'LIVER_FIRM', 'SPLEEN_PALPABLE', 'ASCITES', 'ANOREXIA'])),
    ('clf', RandomForestClassifier())
])

# back to dataframes (for HashingEncoder)
Xtrain = pd.DataFrame(Xtrain)
Xtrain.columns = hepatitis_df.columns
Xtest = pd.DataFrame(Xtest)
Xtest.columns = hepatitis_df.columns

print "RandomForestClassifier Accuracy:", ppl.fit(Xtrain, ytrain).score(Xtest, ytest)