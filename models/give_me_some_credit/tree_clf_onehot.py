from sqlalchemy import create_engine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np 
from scipy import sparse
import pickle
import joblib

import os
print(f"current working directory is: {os.getcwd()}")

class Config:
    engine = create_engine('mysql+pymysql://root:123@localhost:3307/give_me_some_credit')
    train = 'cs_training'
    label_field = 'SeriousDlqin2yrs'
    encoded_train = r".\encoders\give_me_some_credit\cs_training_discrete_onehot.npz"
    train_dict_file = r".\encoders\give_me_some_credit\onehot_encode_dict.pkl"
    # TODO: add log function
    log_file = None
    feature_importance_file = r".\models\give_me_some_credit\feature_importance_onehot.json"

    test = 'cs_test'
    encoded_test = r'.\encoders\give_me_some_credit\cs_test_discrete_onehot.npz'
    


param_grid = {
    'criterion':['gini', 'entropy', 'log_loss'],
    'max_depth':[None, 8, 10, 12, 14],
    'min_samples_split':[2, 8, 14],
    'min_samples_leaf':[400, 450, 500, 600],
    'max_leaf_nodes':[None, 200, 400],
    'random_state':[10,],
    'class_weight':[None, 'balanced',],
    # 'monotonic_cst':np.array([1, 0, -1,...)], 
}

class ModelConfig:
    base_dir = r".\models\give_me_some_credit"
    cv_clf = os.path.join(base_dir, "cv_clf.pkl")
    model_file = os.path.join(base_dir, "tree_clf_onehot.joblib")
    test_metrics = os.path.join(base_dir, "test_metrics_onehot.pkl")
    pred_file = os.path.join(base_dir, "sampleEntryOneHot.csv")




def get_train_sample():
    label = pd.read_sql(sql=f"select {Config.label_field} from {Config.train}", con=Config.engine)
    encoded_data = sparse.load_npz(file=Config.encoded_train)
    x_train, x_test, y_tran, y_test = train_test_split(encoded_data, label, train_size=0.8, random_state=10)
    return x_train, x_test, y_tran, y_test


def get_pred_sample():
    id = pd.read_sql(sql=f"select Id from {Config.test}", con=Config.engine)
    encoded_data = sparse.load_npz(file=Config.encoded_test)
    return id,encoded_data


def trainer():
    x_train, x_test, y_train, y_test = get_train_sample()
    estimator = DecisionTreeClassifier()
    clf = GridSearchCV(estimator=estimator,
                     param_grid=param_grid,
                     scoring='roc_auc',
                     n_jobs=-1,
                     refit=True,
                     cv=5,
                     verbose=3,
                     return_train_score=False
                     )
    clf.fit(x_train, y_train)

    # save cross validate clf
    with open(ModelConfig.cv_clf, 'wb') as f:
        pickle.dump(clf, f)
    
    # best params and socres on validations
    print(f"best params:\n {clf.best_params_} \n {clf.best_score_}")
    print(f"best socres on validations:\n {clf.best_score_}")

    # save best estimator by joblib format
    joblib.dump(clf.best_estimator_, ModelConfig.model_file)

    # test performance 
    test_metrics = {}
    y_test_proba = clf.best_estimator_.predict_proba(x_test)
    proba = list(map(lambda row: row[1], y_test_proba))
    pred = clf.best_estimator_.predict(x_test)

    test_metrics['auc'] = roc_auc_score(y_test, proba)
    test_metrics['accuray'] = accuracy_score(y_test, pred)
    test_metrics['precision'] = precision_score(y_test, pred)
    test_metrics['recall'] = recall_score(y_test, pred)
    test_metrics['f1'] = f1_score(y_test, pred)
    
    with open(ModelConfig.test_metrics, 'wb') as f:
        pickle.dump(test_metrics, f)
    print(f"test metrics: \n {test_metrics}")


def predict():
    id, encoded_data = get_pred_sample()
    m = joblib.load(ModelConfig.model_file)
    proba = m.predict_proba(encoded_data)
    proba = list(map(lambda row: row[1], proba))
    id['Probability'] = proba
    id.to_csv(ModelConfig.pred_file, index=False)
    print(id.head(5))





if __name__ == "__main__":
    trainer()
    predict()