# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:35:23 2018

@author: research
"""

import numpy as np
import pandas as pd
import argparse
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score

classes = ['Walking', 'Jogging', 'Standing', 'Stairs', 'LyingDown', 'Sitting']

def scores_per_class(labels, predicted):
    total = len(labels)
    confusion_mat = np.zeros((6,6))
    for i in range(total):
        lab_index = classes.index(labels.iloc[i])
        pred_index = classes.index(predicted[i])
        confusion_mat[lab_index][pred_index] += 1
    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted, average='macro')
    recall = recall_score(labels, predicted, average='macro')
    f1 = f1_score(labels,predicted,average='macro')
    return (confusion_mat, accuracy,precision, recall, f1)

def normal_cross_val(Features, Labels, test_fold_no, model):
    #10 folds exist in the features and Lables already and test fold number is the number ot test with
    X_test = Features[test_fold_no]
    y_test = Labels[test_fold_no]
    frames = []
    lab_frames = []
    for i in range(10):
        if i==test_fold_no:
            continue
        else:
            frames.append(Features[i])
            lab_frames.append(Labels[i])
    X_train = pd.concat(frames)
    y_train = pd.concat(lab_frames)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return scores_per_class(y_test, y_pred)

def balanced_cross_val(Features, Labels, test_fold_no, model, AR_data):
    #10 folds exist in the features and Lables already and test fold number is the number ot test with
    X_test = Features[test_fold_no]
    y_test = Labels[test_fold_no]
    frames = []
    lab_frames = []
    for i in range(10):
        if i==test_fold_no:
            continue
        else:
            frames.append(Features[i])
            lab_frames.append(Labels[i])
    X_train = pd.concat(frames)
    y_train = pd.concat(lab_frames)
    #find the class proportions in this
    proportions = []
    for c in classes:
        proportions.append(sum([i==c for i in y_train]))
    total_training = len(y_train)
    print total_training
    target_total = max(proportions)
    for i in range(len(classes)):
        if proportions[i]<target_total:
            AR_class = AR.loc[AR['Class']==classes[i]]
            if not AR_class.empty:
                no_new_samples = target_total - proportions[i]
                #print no_new_samples, " to be added"
                if AR_class.shape[0]>no_new_samples:
                    sampled_AR = AR_class.sample(n = no_new_samples)
                else:
                    sampled_AR = AR_class
                X_train = pd.concat([X_train, sampled_AR[sampled_AR.columns[1:44]]])
                y_train = pd.concat([y_train, sampled_AR[sampled_AR.columns[44]]])
        print classes[i]," has ",sum([j==classes[i] for j in y_train]), " training samples."
    model.fit(X_train, y_train)
    print "Testing samples toatl:", len(y_test)
    y_pred = model.predict(X_test)
    return scores_per_class(y_test, y_pred)

parser = argparse.ArgumentParser()
parser.add_argument('tx_train', type =str, help = 'train transformed output file')
parser.add_argument('additional_train', type = str, help = 'additional AR transformed output files')
args = parser.parse_args()

act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Class', 'Time']

feats = pd.read_csv(args.tx_train, header = None, names=act_headers)
AR = pd.read_csv(args.additional_train, header = None, names = act_headers)
#replacing Upstair and Downstairs with Stairs
AR = AR.replace(to_replace=['Upstairs','Downstairs'], value = 'Stairs')
print AR['Class'].unique()
'''
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(feats[feats.columns[1:44]], feats[feats.columns[44]])

'''
data = shuffle(feats)
no_of_samples = data.shape[0]
samples_per_fold = no_of_samples/10
Features_10_folds = []
Labels_10_folds = []

for i in range(9):
    data_fold = data[:samples_per_fold]
    data = data[samples_per_fold:]
    features = data_fold[data_fold.columns[1:44]]
    labels = data_fold[data_fold.columns[44]]
    Features_10_folds.append(features)
    Labels_10_folds.append(labels)

#for last fold all remaining
features = data[data.columns[1:44]]
labels = data[data.columns[44]]
Features_10_folds.append(features)
Labels_10_folds.append(labels)

print Features_10_folds[0].shape

clf = RandomForestClassifier(max_depth = 8, random_state=0)
mlp = MLPClassifier(hidden_layer_sizes = (50,25))
gbt = GradientBoostingClassifier()
ovr = OneVsRestClassifier(RandomForestClassifier())
eclf = VotingClassifier(estimators=[('gbt',gbt),('ovr',ovr)], voting='soft')
acc_RF = []
prec_RF = []
rec_RF = []
f1_RF = []
acc_MLP = []
prec_MLP = []
rec_MLP = []
f1_MLP = []
acc_GBT = []
prec_GBT = []
rec_GBT = []
f1_GBT = []
acc_ovr = []
prec_ovr = []
rec_ovr = []
f1_ovr = []
acc_ECLF = []
prec_ECLF = []
rec_ECLF = []
f1_ECLF=  []
acc_RF_aug = []
acc_MLP_aug = []
acc_GBT_aug = []
acc_ovr_aug = []
acc_ECLF_aug = []
for i in range(10):
    print "cross val results for set ", str(i)
    print "Random forest:"
    conf, a, p, r ,f = normal_cross_val(Features_10_folds, Labels_10_folds, i, clf)
    acc_RF.append(a)
    prec_RF.append(p)
    rec_RF.append(r)
    f1_RF.append(f)
    print conf
    print "Accuracy:", a
    print "MLP:"
    conf, a, p ,r, f = normal_cross_val(Features_10_folds, Labels_10_folds, i, mlp)
    acc_MLP.append(a)
    prec_MLP.append(p)
    rec_MLP.append(r)
    f1_MLP.append(f)
    print conf
    print "Accuracy:", a
    print "Gradient Boosted Trees:"
    conf, a, p ,r, f = normal_cross_val(Features_10_folds, Labels_10_folds, i, gbt)
    acc_GBT.append(a)
    prec_GBT.append(p)
    rec_GBT.append(r)
    f1_GBT.append(f)
    print conf
    print "Accuracy:", a
    print "Linear Classifier with ovr training"
    conf, a, p, r, f = normal_cross_val(Features_10_folds, Labels_10_folds, i , ovr)
    acc_ovr.append(a)
    prec_ovr.append(p)
    rec_ovr.append(r)
    f1_ovr.append(f)
    print conf
    print "Accuracy:", a
    print "Voting Clf for GBT and ovr:"
    conf, a, p, r, f = normal_cross_val(Features_10_folds, Labels_10_folds, i, eclf)
    acc_ECLF.append(a)
    prec_ECLF.append(p)
    rec_ECLF.append(r)
    f1_ECLF.append(f)
    print conf
    print "Accuracy:", a
    print "RF with AR augumented:"
    conf, ai, _,_,_ = balanced_cross_val(Features_10_folds, Labels_10_folds, i, clf, AR)
    acc_RF_aug.append(a)
    print conf
    print "Accuracy:", a
    print "MLP with AR augumented:"
    conf, a,_,_,_ = balanced_cross_val(Features_10_folds, Labels_10_folds, i, mlp, AR)
    acc_MLP_aug.append(a)
    print conf
    print "Accuracy:", a
    print "GBT with augumented AR:"
    conf, a,_,_,_ = balanced_cross_val(Features_10_folds, Labels_10_folds, i, gbt, AR)
    acc_GBT_aug.append(a)
    print conf
    print "Accuracy:", a
    print "Linear Classifier with ovr training and augumented AR"
    conf, a,_,_,_ = balanced_cross_val(Features_10_folds, Labels_10_folds, i, ovr, AR)
    acc_ovr_aug.append(a)
    print conf
    print "Accuracy:", a
    print "Voting Classifier with GBT and ovr and augumented AR"
    conf, a,_,_,_ = balanced_cross_val(Features_10_folds, Labels_10_folds, i, eclf, AR)
    acc_ECLF_aug.append(a)
    print conf
    print "Accuracy:", a

print "Average accuracy scores:"
print "RF:", sum(acc_RF)/10.0
print "MLP:", sum(acc_MLP)/10.0
print "GBT:", sum(acc_GBT)/10.0
print "ovr:", sum(acc_ovr)/10.0
print "Voting", sum(acc_ECLF)/10.0
print "RF augumented:", sum(acc_RF_aug)/10.0
print "MLP augumented:", sum(acc_MLP_aug)/10.0
print "GBT augumented:", sum(acc_GBT_aug)/10.0
print "ovr augumented:", sum(acc_ovr_aug)/10.0
print "Voting augumented:", sum(acc_ECLF_aug)/10.0

print " classifiers "," Accuracy "," Precision "," Recall "," F1 score "
print "Random Forest:", np.mean(acc_RF),"(",np.std(acc_RF),")", " ",np.mean(prec_RF),"(",np.std(prec_RF),")", " ",np.mean(rec_RF),"(",np.std(rec_RF),")"," ",np.mean(f1_RF),"(",np.std(f1_RF),")"
print "MLP:", np.mean(acc_MLP),"(",np.std(acc_MLP),")", " ",np.mean(prec_MLP),"(",np.std(prec_MLP),")", " ",np.mean(rec_MLP),"(",np.std(rec_MLP),")"," ",np.mean(f1_MLP),"(",np.std(f1_MLP),")"
print "Gradient Boosted Trees:", np.mean(acc_GBT),"(",np.std(acc_GBT),")", " ",np.mean(prec_GBT),"(",np.std(prec_GBT),")", " ",np.mean(rec_GBT),"(",np.std(rec_GBT),")"," ",np.mean(f1_GBT),"(",np.std(f1_GBT),")"
print "One vs Rest with RF:", np.mean(acc_ovr),"(",np.std(acc_ovr),")", " ",np.mean(prec_ovr),"(",np.std(prec_ovr),")", " ",np.mean(rec_ovr),"(",np.std(rec_ovr),")"," ",np.mean(f1_ovr),"(",np.std(f1_ovr),")"
print "Voting clf (GBT, OVR):", np.mean(acc_ECLF),"(",np.std(acc_ECLF),")", " ",np.mean(prec_ECLF),"(",np.std(prec_ECLF),")", " ",np.mean(rec_ECLF),"(",np.std(rec_ECLF),")"," ",np.mean(f1_ECLF),"(",np.std(f1_ECLF),")"

Feature = feats[feats.columns[1:44]]
Label = feats[feats.columns[44]]
ovr.fit(Feature, Label)
eclf.fit(Feature, Label)
print "Saving OVR classifier as OVR.pickle"
joblib.dump(ovr, 'OVR.pickle')
print "Saving Voting classifier with OVR and GBT soft voting using argmax into Voting.pickle"
joblib.dump(eclf, 'Voting.pickle')

#cross_scores = cross_val_score(clf, feats[feats.columns[1:44]], feats[feats.columns[44]], cv=10)
#print cross_scores
#clf.fit(X_train, y_train)

#Out[20]:
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=2, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#            oob_score=False, random_state=0, verbose=0, warm_start=False)
'''
print clf.score(X_test, y_test)
#Out[21]: 0.9473684210526315
print "Now training on whole of this and test on test file"
feats_test = pd.read_csv(args.tx_test, header=None, names = act_headers)
X_train = feats[feats.columns[1:44]]
y_train = feats[feats.columns[44]]
X_test = feats_test[feats_test.columns[1:44]]
y_test = feats_test[feats_test.columns[44]]
clf.fit(X_train, y_train)
print y_train.unique(), y_test.unique()
print "Test score:", clf.score(X_test, y_test)

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
