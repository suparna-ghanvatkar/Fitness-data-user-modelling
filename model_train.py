# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:35:23 2018

@author: research
"""

import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('tx', type =str, help = 'transformed output file')
args = parser.parse_args()

act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Class', 'Time']

feats = pd.read_csv(args.tx, header = None, names=act_headers)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(feats[feats.columns[1:44]], feats[feats.columns[44]])

'''
X_train, X_test, y_train, y_test= train_test_split(feats[feats.columns[1:44]], feats[feats.columns[44]], test_size=0.20)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

#Out[20]: 
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=2, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#            oob_score=False, random_state=0, verbose=0, warm_start=False)

clf.score(X_test, y_test)
#Out[21]: 0.9473684210526315


X_train, X_test, y_train, y_test= train_test_split(feats[feats.columns[1:44]], feats[feats.columns[44]], test_size=0.20, random_state = 42)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

Out[30]: 
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

clf.score(X_test, y_test)
Out[31]: 0.9298245614035088
'''