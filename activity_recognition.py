'''
This script takes a transformed file as input, and also the model file saved using joblib and produces new file with activity labels.
'''

import argparse
import numpy as np
import pandas as pd
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument('tx', type = str, help = 'transformed featured file input path')
parser.add_argument('model', type = str, help = 'model file path saved using joblib')
args = parser.parse_args()

act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Class', 'Time']

feats = pd.read_csv(args.tx, header = None, names=act_headers)
clf = joblib.load(args.model)

pred_labels = clf.predict(feats[feats.columns[1:44]])
feats['Class'] = list(pred_labels)

feats.to_csv(args.tx+'_predicted.txt', header=False, index=False)
