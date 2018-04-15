'''
Script to perform user clustering and frequent pattern mining given user activity data and demographics
'''

import argparse
import numpy as np
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('act', type=str, help='activity file path')
parser.add_argument('demo',type=str,help='demographics file path')
args = parser.parse_args()

act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Activity', 'Time']
demo_headers = ['user','height','gender','age','weight','leg_injury']

activity_data = pd.read_csv(args.act, header=None, names=act_headers)
demographics = pd.read_csv(args.demo, header=None, names=demo_headers)
activity_data['time'] = pd.to_datetime(activity_data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

activity_encoding = {'Walking':'A', 'Jogging':'B', 'Stairs':'C', 'Sitting':'D', 'Standing':'E', 'LyingDown':'F'}
#generating user timeline..and storing
users = activity_data.user.unique()

uact = {}
ugaps = {}
durations = []
for usr in users:
    uact[usr] = {}
    ugaps[usr] = []

for usr in users:
    act_seq = []
    user_act_data = activity_data.loc[activity_data.user==usr]['Activity','Time']
    user_act_data = user_act_data.sort_values(by='Time')
    prev = user_act_data.iloc[0]['Activity']
    prev_time = user_act_data.iloc[0]['Time']
    stime = prev_time
    gstime = stime
    for i, row in user_act_data.iterrows():
        if row['Activity']!=prev:
            etime = prev_time
            uact[usr][stime] = [etime, (etime-stime).total_seconds()//60, prev]
            stime = row['Time']
            gap = (stime-etime).total_seconds()
            print usr, gap, etime, stime
            if gap>120:
                ugaps[usr].append(gap)
            prev = row['Activity']
            prev_time = stime
        elif (prev_time-row['Time']).total_seconds()>120:
            etime = prev_time
            uact[usr][stime] = [etime, (etime-stime).total_seconds()//60, prev]
            stime = row['Time']
            gap = (stime-etime).total_seconds()
            print usr, gap, etime, stime
            ugaps[usr].append(gap)
            prev_time = stime
        else:
            prev_time = row['Time']
        getime = prev_time
    durations.append((getime-gstime).total_seconds()//60)
print uact
print ugaps

pickle.dump(durations, open('durations.pickle','wb'))
pickle.dump(uact,open('userwise_activity.pickle', 'wb'))
pickle.dump(ugaps, open('userwise_gaps.pickle','wb'))

print "User durations:"
for i,dur in durations.enumerate():
    print users[i], dur

