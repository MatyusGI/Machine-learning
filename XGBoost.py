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
x_train
y_test
x_test

label_counts = y_train.value_counts()
active = float(label_counts[1])
inactive = float(label_counts[0])
wpos = inactive/active
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
model = XGBClassifier(scale_pos_weight = wpos)
kfold = KFold(n_splits = 7)
results = cross_val_score(model, x_train, y_train, cv = kfold)
print('accuracy: %.3f%% (%.3f%%)' % (results.mean()*100, results.std()*100))

model = XGBClassifier(scale_pos_weight = wpos)
model.fit(x_train, y_train)
for feature, importance in zip(x_train, model.feature_importances_):
    print(feature, importance*100.0)
