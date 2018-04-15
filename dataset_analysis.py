import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('data', type =str, help = 'train transformed output file')
args = parser.parse_args()

act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Class', 'Time']
dataset = pd.read_csv(args.data, header = None, names = act_headers)

diff_activities = dataset.Class.unique()
for act in diff_activities:
    act_data = dataset.loc[dataset.Class==act]
    print act, ":", act_data.shape[0]

#for every user, list all activities:
users = dataset.user.unique()
for usr in users:
    user_data = dataset.loc[dataset.user==usr]
    print "----"
    print "for user:", usr
    for act in diff_activities:
        usr_act_data = user_data.loc[user_data.Class==act]
        print act, ":", usr_act_data.shape[0],
    print "Total activity log:", user_data.shape[0]*10/60 , "mins"
