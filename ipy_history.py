# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:50:42 2018

@author: research
"""

# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Thu Mar 15 14:37:36 2018)---
import numpy as np
import pandas as pd
import os

datadir = os.path.join('..', 'WISDM_at_v2.0')
unlabelled_act = os.path.join(datadir,'WISDM_at_v2.0_unlabeled_transformed.csv')
unlabelled_raw = os.path.join(datadir, 'WISDM_at_v2.0_unlabeled_raw.txt')
raw_headers = ['user','activity','time','x','y','z']
act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Activity']
raw_data = pd.read_csv(unlabelled_raw, header=None, names = raw_headers)
act_data = pd.read_csv(unlabelled_act, header=None, names = act_headers)

time_user_raw = raw_data[['user','time']]
user_duration = time_user_raw.groupby('user').agg({'time':[min,max]})
user_duration.columns = ["_".join(x) for x in user_duration.columns.ravel()]
user_duration['duration'] = user_duration['time_max'] - user_duration['time_min']
user_duration.sort_values(by='duration')

import numpy as np
import pandas as pd
import os

datadir = os.path.join('..', 'WISDM_at_v2.0')
unlabelled_act = os.path.join(datadir,'WISDM_at_v2.0_unlabeled_transformed.csv')
unlabelled_raw = os.path.join(datadir, 'WISDM_at_v2.0_unlabeled_raw.txt')
raw_headers = ['user','activity','time','x','y','z']
act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Activity']
raw_data = pd.read_csv(unlabelled_raw, header=None, names = raw_headers)
act_data = pd.read_csv(unlabelled_act, header=None, names = act_headers)

time_user_raw = raw_data[['user','time']]
user_duration = time_user_raw.groupby('user').agg({'time':[min,max]})
user_duration.columns = ["_".join(x) for x in user_duration.columns.ravel()]
user_duration['duration'] = user_duration['time_max'] - user_duration['time_min']
user_duration.sort_values(by='duration')

raw_data['user'].unique()
len(raw_data['user'].unique())
len(act_data['user'].unique())
labelled_act = os.path.join(datadir,'WISDM_at_v2.0_transformed.csv')
labelled_raw = os.path.join(datadir, 'WISDM_at_v2.0_raw.txt')
lraw_data = pd.read_csv(labelled_raw, header=None, names = raw_headers)
lact_data = pd.read_csv(labelled_act, header=None, names = act_headers)

len(lact_data['user'].unique())
len(lraw_data['user'].unique())
len(concat(lraw_data['user'],raw_data['user']).unique())
len(pd.concat(lraw_data['user'],raw_data['user']).unique())
len(pd.concat([lraw_data['user'],raw_data['user']]).unique())
len(pd.concat([lact_data['user'],act_data['user']]).unique())
act_data.head()

act_user['user'=194]
act_user['user'==194]
act_dat['user'=194]
act_data['user'=194]
act_data['user'==194]
act_data['user']==194
act_data.iloc[act_data['user']==194]
act_data.loc[act_data['user']==194]
act_data.loc[act_data['user']==194]['Activity']
act_data['Activity']
user_act_data = act_data.loc[act_data['user']==194]['Activity']
for row in user_act_data.iteritems():
    print row
    
user_act_data[0]
start = user_act_data[0]
prev = start
for i,activity in user_act_data.iteritems():
    if activity!=prev:
        print "Transition from",prev," to ",activity
    prev = activity
    
act_data.groupby('user').count()
act_data.groupby('user').count()[activity]
act_data.groupby('user').count()['Activity']
len(act_data['user'].unique())
len(pd.concat([lact_data['user'],act_data['user']]).unique())
len(lact_data['user'].unique())
act_data.columns
user_inst = act_data.groupby('user').count()['Activity']
user_inst.plot()
user_inst['Activity'].min()
user_inst.head()
user_inst.min()
user_inst.sortby('Activity')
user_inst.sort()
user_inst.sort_values()