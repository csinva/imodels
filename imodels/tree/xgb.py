from sklearn import datasets
from xgboost import XGBRegressor, XGBClassifier

if __name__ == '__main__':
    X, y = datasets.load_diabetes(return_X_y=True)
    xgb = XGBRegressor()
    xgb.fit(X, y)
    trees = xgb.get_booster().get_dump()

    xgb.predict(X)
