'''
This script filters users according to ones having demographics and more than 4 hours of data
'''

import pickle
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('activity', type = str, help = 'path to activity file of all users')
parser.add_argument('demo', type=str, help = 'path to demographics file')
args = parser.parse_args()

#datadir = os.path.join('..', 'WISDM_at_v2.0')
#raw_headers = ['user','activity','time','x','y','z']
act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Activity', 'Time']
demo_headers = ['user','height','gender','age','weight','leg_injury']

'''
labelled_act = os.path.join(datadir,'WISDM_at_v2.0_transformed.csv')
labelled_raw = os.path.join(datadir, 'WISDM_at_v2.0_raw.txt')
unlabelled_act = os.path.join(datadir,'WISDM_at_v2.0_unlabeled_transformed.csv')
unlabelled_raw = os.path.join(datadir, 'WISDM_at_v2.0_unlabeled_raw.txt')
unlab_raw = pd.read_csv(unlabelled_raw, header=None, names = raw_headers, parse_dates=True)
unlab_act = pd.read_csv(unlabelled_act, header=None, names = act_headers)
lab_raw = pd.read_csv(labelled_raw, header=None, names = raw_headers, parse_dates=True)
lab_act = pd.read_csv(labelled_act, header=None, names = act_headers)
demo = os.path.join(datadir, 'WISDM_at_v2.0_demographics.txt')
demographics = pd.read_csv(demo, header=None, names = demo_headers)
'''
lab_act = pd.read_csv(args.activity, header=None, names = act_headers, parse_dates=True)
demographics = pd.read_csv(args.demo, header=None, names = demo_headers)

print lab_act[lab_act.Time=='0.0']
#lab_raw = lab_raw.loc[lab_raw['time']>0]
lab_act['Time'] = pd.to_datetime(lab_act['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
#unlab_raw = unlab_raw.loc[unlab_raw['time']>0]
#unlab_raw['time'] = pd.to_datetime(unlab_raw['time'], unit = 'ms', errors='coerce')

#users = list(set(unlab_raw.user.unique()).intersection(set(unlab_act.user.unique())))
users = list(set(lab_act.user.unique()).intersection(set(demographics.user.unique())))
print len(users)
time_user_raw = lab_act[['user','Time']]
user_duration = time_user_raw.groupby('user').agg({'Time':[min,max]})
user_duration.columns = ["_".join(x) for x in user_duration.columns.ravel()]
print user_duration.head()
user_duration['duration'] = user_duration['Time_max'] - user_duration['Time_min']
#print user_duration.sort_values(by='duration').loc[unlab_act.user.unique()]['duration']
user_duration = user_duration.loc[users]
user_duration = user_duration[(user_duration['duration'].dt.total_seconds()) > 10800]
#users in labelled set
#lab_users = list(set(lab_raw.user.unique()).intersection(set(lab_act.user.unique())))
#common users
#comm_users = list(set(lab_users).intersection(set(users)))
'''
time_user_raw_l = lab_raw[['user','time']]
time_user_raw_l = time_user_raw_l.loc[time_user_raw_l['user'].isin(comm_users)]
user_duration_l = time_user_raw_l.groupby('user').agg({'time':[min,max]})
user_duration_l.columns = ["_".join(x) for x in user_duration_l.columns.ravel()]
user_duration_l['duration'] = user_duration_l['time_max'] - user_duration_l['time_min']
user_duration_l = user_duration_l[(user_duration_l['duration'].dt.total_seconds()) > 10800]

all_users = users
users = [usr for usr in all_users if (usr in demographics.user.unique()) and ((usr in user_duration.index) or (usr in user_duration_l.index))]
print len(users)
'''
#users with enough data duration:
users = user_duration.index
print len(users)

#users with more than 4 hours data and demographics available are 23
#dumping their raw and activity data and demographics data
#filtered_raw_lab = lab_raw.loc[lab_raw['user'].isin(users)]
filtered_act_lab = lab_act.loc[lab_act['user'].isin(users)]
filtered_demo = demographics.loc[demographics['user'].isin(users)]
#filtered_raw_unlab = unlab_raw.loc[unlab_raw['user'].isin(users)]
#filtered_act_unlab = unlab_act.loc[unlab_act['user'].isin(users)]
#pickle.dump(filtered_raw_lab, open('filtered_raw_lab.csv','wb'))
pickle.dump(filtered_act_lab, open('filtered_user_act.csv','wb'))
#pickle.dump(filtered_raw_unlab, open('filtered_raw_unlab.csv','wb'))
#pickle.dump(filtered_act_unlab, open('filtered_act_unlab.csv','wb'))
pickle.dump(filtered_demo, open('filtered_user_demo.csv','wb'))
