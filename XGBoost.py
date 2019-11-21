import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
y_tr = pd.read_csv('tox21_labels_train.csv.gz', index_col=0, compression="gzip")
y_te = pd.read_csv('tox21_labels_test.csv.gz', index_col=0, compression="gzip")
x_tr = pd.read_csv('tox21_dense_train.csv.gz', index_col=0, compression="gzip")
x_te = pd.read_csv('tox21_dense_test.csv.gz', index_col=0, compression="gzip")

rows_tr = np.isfinite(y_tr['SR.p53']).values
rows_te = np.isfinite(y_te['SR.p53']).values
y_train = y_tr['SR.p53'][rows_tr]
x_train = x_tr[rows_tr]
y_test = y_te['SR.p53'][rows_te]
x_test = x_te[rows_te]

y_train
