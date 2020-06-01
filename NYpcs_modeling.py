# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
from xgboost import XGBRegressor
import category_encoders as ce
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pdpbox.pdp import pdp_interact, pdp_interact_plot, pdp_isolate, pdp_plot
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.impute import SimpleImputer
import sklearn.metrics as metrics
from xgboost import XGBClassifier, plot_importance
import uuid

url = "https://media.githubusercontent.com/media/qweliant/PCSstudy/master/data/pcs.csv"
pcs = pd.read_csv(url)
pcs.drop("survey_year", axis=1, inplace=True)

pcs["id"] = [uuid.uuid1() for k in pcs.index]


target = "serious_mental_illness"
X = pd.DataFrame(
    pcs.drop(
        columns=[
            target,
            "mental_illness",
            "principal_diagnosis_class",
            "additional_diagnosis_class",
        ]
    )
)
y = pd.DataFrame(pcs[target])


X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.001, train_size=0.10, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.05,
    train_size=0.10,
    stratify=y_trainval,
    random_state=42,
)


train_id = X_train["id"]
val_id = X_val["id"]
test_id = X_test["id"]

X_train = X_train.drop("id", axis=1)
X_val = X_val.drop("id", axis=1)
X_test = X_test.drop("id", axis=1)


x_processor = make_pipeline(ce.OrdinalEncoder(), SimpleImputer(strategy="median"))
y_processor = make_pipeline(ce.OrdinalEncoder(), SimpleImputer(strategy="median"))

cols = X_train.columns
len(cols)


def prepare_inputs(X_train, X_val, X_test):
    X_train_enc = pd.DataFrame(x_processor.fit_transform(X_train), columns=cols)
    X_val_enc = pd.DataFrame(x_processor.transform(X_val), columns=cols)
    X_test_enc = pd.DataFrame(x_processor.transform(X_test), columns=cols)
    return X_train_enc, X_val_enc, X_test_enc


def prepare_targets(y_train, y_val, y_test):
    y_train_enc = pd.DataFrame(
        y_processor.fit_transform(y_train), columns=["serious_mental_illness"]
    )
    y_val_enc = pd.DataFrame(
        y_processor.transform(y_val), columns=["serious_mental_illness"]
    )
    y_test_enc = pd.DataFrame(
        y_processor.transform(y_test), columns=["serious_mental_illness"]
    )
    return y_train_enc, y_val_enc, y_test_enc


X_train_processed, X_val_processed, X_test = prepare_inputs(X_train, X_val, X_test)

y_train_processed, y_val_processed, y_test_processed = prepare_targets(
    y_train, y_val, y_test
)

num_of_classes = len(y.serious_mental_illness.unique())

eval_set = [(X_train_processed, y_train_processed), (X_val_processed, y_val_processed)]

xgbcl = XGBClassifier(
    base_score=0.5,
    booster="gbtree",
    colsample_bylevel=1.0,
    gamma=0.0,
    max_delta_step=0.0,
    min_child_weight=1.0,
    num_class=num_of_classes,
    missing=None,
    n_jobs=-1,
    objective="multi:softprob",
    random_state=42,
    reg_alpha=0.0,
    reg_lambda=1.0,
    scale_pos_weight=1.0,
    tree_method="auto",
)

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {
    "colsample_bytree": [0.75, 1],
    "learning_rate": [0.01, 0.05, 0.1, 0.3, 0.5],
    "max_depth": [1, 2, 3, 5],
    "subsample": [0.75, 1],
    "n_estimators": list(range(50, 400, 50)),
}

grid_search = GridSearchCV(estimator=xgbcl, param_grid=param_grid, n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(
    X_train_processed, y_train_processed, eval_metric="mlogloss"
)

xgbcl = XGBClassifier(
    base_score=0.5,
    booster="gbtree",
    colsample_bylevel=1.0,
    gamma=0.0,
    max_delta_step=0.0,
    min_child_weight=1.0,
    missing=None,
    n_jobs=-1,
    objective="multi:softprob",
    num_class=num_of_classes,
    random_state=42,
    reg_alpha=0.0,
    reg_lambda=1.0,
    scale_pos_weight=1.0,
    tree_method="auto",
    colsample_bytree=grid_result.best_params_["colsample_bytree"],
    learning_rate=grid_result.best_params_["learning_rate"],
    max_depth=grid_result.best_params_["max_depth"],
    subsample=grid_result.best_params_["subsample"],
    n_estimators=grid_result.best_params_["n_estimators"],
    eval_metric="auc",
)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# refit the model on k-folds to get stable avg error metrics
scores = cross_validate(
    estimator=xgbcl, X=X_train_processed, y=y_train_processed, cv=kfold, n_jobs=-1
)


import pickle

xgbcl.fit(X_train_processed, y_train_processed)


# Generate predictions against our training and test data
pred_train = xgbcl.predict(X_train_processed)
proba_train = xgbcl.predict_proba(X_train_processed)
pred_val = xgbcl.predict(X_val_processed)
proba_val = xgbcl.predict_proba(X_val_processed)
pred_test = xgbcl.predict(X_test)
proba_test = xgbcl.predict_proba(X_test)
