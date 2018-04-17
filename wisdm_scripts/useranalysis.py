# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:32:18 2018

@author: suparna
"""

import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('raw',type=str, help='activity data file' )
args = parser.parse_args()


'''datadir = os.path.join('..', 'WISDM_at_v2.0')
labelled_act = os.path.join(datadir,'WISDM_at_v2.0_transformed.csv')
labelled_raw = os.path.join(datadir, 'WISDM_at_v2.0_raw.txt')
unlabelled_act = os.path.join(datadir,'WISDM_at_v2.0_unlabeled_transformed.csv')
unlabelled_raw = os.path.join(datadir, 'WISDM_at_v2.0_unlabeled_raw.txt')
demo = unlabelled_raw = os.path.join(datadir, 'WISDM_at_v2.0_demographics.txt')
'''

raw_headers = ['user','activity','time','x','y','z']
act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Activity', 'time']
demo_headers = ['user','height','gender','age','weight','leg_injury']
raw_data = pd.read_csv(args.raw,header=None, names=act_headers)
raw_data['time'] = pd.to_datetime(raw_data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
users = raw_data.user.unique()
'''
unlab_raw = pd.read_csv(unlabelled_raw, header=None, names = raw_headers, parse_dates=True)
unlab_act = pd.read_csv(unlabelled_act, header=None, names = act_headers)
lab_raw = pd.read_csv(labelled_raw, header=None, names = raw_headers, parse_dates=True)
lab_act = pd.read_csv(labelled_act, header=None, names = act_headers)
demographics = pd.read_csv(demo, header=None, names = demo_headers)

#convert to time
lab_raw = lab_raw.loc[lab_raw['time']>0]
lab_raw['time'] = pd.to_datetime(lab_raw['time'], unit = 'ms', errors='coerce')
unlab_raw = unlab_raw.loc[unlab_raw['time']>0]
unlab_raw['time'] = pd.to_datetime(unlab_raw['time'], unit = 'ms', errors='coerce')

raw_data = pd.concat([unlab_raw,lab_raw])
act_data = pd.concat([unlab_act, lab_act])

users = list(set(raw_data.user.unique()).intersection(set(act_data.user.unique())))
print len(users)
time_user_raw = raw_data[['user','time']]
user_duration = time_user_raw.groupby('user').agg({'time':[min,max]})
user_duration.columns = ["_".join(x) for x in user_duration.columns.ravel()]
user_duration['duration'] = user_duration['time_max'] - user_duration['time_min']
#print user_duration.sort_values(by='duration').loc[act_data.user.unique()]['duration']
user_duration = user_duration.loc[users]
user_duration = user_duration[(user_duration['duration'].dt.total_seconds()) > 14400]

#print user_duration.columns
users = user_duration.index.values.tolist()
print len(users)
raw_data = raw_data.loc[raw_data['user'].isin(users)]
act_data = act_data.loc[act_data['user'].isin(users)]

demo_users = demographics.user.unique()
raw_data = raw_data.loc[raw_data['user'].isin(demo_users)]
act_data = act_data.loc[act_data['user'].isin(demo_users)]
comm_users = list(set(lab_raw.user.unique()).intersection(set(users)))  #check specifically for these users

import pickle
pickle.dump(raw_data, open('raw_data_selected.dump','wb'))
pickle.dump(act_data, open('act_data_selected.dump','wb'))
print "raw and act files writen"
'''
ugaps = {}
for userid in users:
    gap_stats = []
    maxgap = 0
    mingap = 0
    sumgap = 0
    user_raw_data = raw_data.loc[raw_data['user']==userid]
    user_raw_data = user_raw_data.sort_values('time')
    if not user_raw_data.empty:
        prev_time = user_raw_data.iloc[0]['time']
        gap = 0
        print userid, 'GAP LENGTHS'
        for index, attr in user_raw_data.iterrows():
            tgap = (attr['time'] - prev_time).total_seconds()
            if  tgap > 120:
                gap = gap + 1
                #print tgap//60, 'min..i.e.', tgap,'s'
                tgap_mins = tgap//60
                if tgap_mins<mingap:
                    mingap = tgap_mins
                elif tgap_mins>maxgap:
                    maxgap = tgap_mins
                sumgap += tgap_mins
            prev_time = attr['time']
        meangap = sumgap/gap
        ugaps[userid] = [mingap,maxgap,meangap,gap]

print ugaps
