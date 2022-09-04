import xgboost as xgb
import numpy as np
import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score


def main():
    
    dataset_name = 'credit-g'
    dataframe, target, features = fetch_data.get_df(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(dataframe[features], dataframe[target], random_state=42, stratify=dataframe[target])

    # xgboost model 
    clf_xgb = xgb.XGBClassifier(objective='binary:logistic', missing=1, seed=42)
    clf_xgb.fit(X_train,
                y_train,
                verbose=True,
                early_stopping_rounds=10,
                eval_metric='aucpr',
                eval_set=[(X_test, y_test)])
    
    # make predictions for test data

    y_pred = clf_xgb.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

if __name__ == "__main__":
    main()