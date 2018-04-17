'''
Script to perform user clustering and frequent pattern mining given user activity data and demographics
'''

import argparse
import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
from pyprefixspan import pyprefixspan
from pymining import seqmining

parser = argparse.ArgumentParser()
parser.add_argument('act', type=str, help='activity file path')
parser.add_argument('demo',type=str,help='demographics file path')
args = parser.parse_args()

act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Activity', 'Time']
demo_headers = ['user','height','gender','age','weight','leg_injury']

activity_data = pd.read_csv(args.act, header=None, names=act_headers)
demographics = pd.read_csv(args.demo, header=None, names=demo_headers)
activity_data['Time'] = pd.to_datetime(activity_data['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

activity_encoding = {'Walking':'A', 'Jogging':'B', 'Stairs':'C', 'Sitting':'D', 'Standing':'E', 'LyingDown':'F'}
#generating user timeline..and storing
users = activity_data.user.unique()

act_seqs = {}
uact = {}
ugaps = {}
durations = []
for usr in users:
    uact[usr] = {}
    ugaps[usr] = []
    act_seqs[usr] = []

for usr in users:
    user_act_data = activity_data.loc[activity_data.user==usr][['Activity','Time']]
    user_act_data = user_act_data.sort_values(by='Time')
    prev_act = user_act_data.iloc[0]['Activity']
    prev_time = user_act_data.iloc[0]['Time'] - timedelta(seconds=10)
    stime = prev_time - timedelta(seconds=10)
    prev_date = prev_time.date()
    gstime = stime
    #seq = activity_encoding[prev]
    print "-----------------------------------------------------------------------------------------------------"
    print "USER ID:",usr
    print "Demographics:"
    print demographics.loc[demographics.user==usr]
    print "Total no of activities recorded:", user_act_data.shape[0]
    print "Total duration of recorded activity:", user_act_data.shape[0]*10/60, "mins"
    for act in activity_encoding.keys():
        act_data = user_act_data.loc[user_act_data.Activity==act]
        print act, ":", act_data.shape[0]*10/60,"mins",
    print "Timeline:."
    per_day_usr_activity = []
    for i, row in user_act_data.iterrows():
        print row
        curr_time = row['Time']
        curr_act = row['Activity']
        curr_date = curr_time.date()
        if curr_date!=prev_date:
            print prev_act, prev_time-stime, " ", stime," ", prev_time
            per_day_usr_activity.append([usr, prev_date, prev_act,stime,prev_time])
            print curr_date
            stime = curr_time - timedelta(seconds=10)
        elif curr_time-prev_time>timedelta(seconds=15):
            print prev_act, prev_time-stime," ", stime, " ", prev_time
            per_day_usr_activity.append([usr, prev_date, prev_act,stime,prev_time])
            print "gap", curr_time-prev_time, " ", prev_time, " ", curr_time
            stime = curr_time - timedelta(seconds=10)
        elif curr_act!=prev_act:
            print prev_act, prev_time-stime, " ", stime, " ", prev_time
            per_day_usr_activity.append([usr, prev_date, prev_act,stime,prev_time])
            stime = curr_time - timedelta(seconds=10)
        prev_time = curr_time
        prev_date = prev_time.date()
        prev_act = curr_act
        '''
        if row['Activity']!=prev:
            etime = prev_time
            #uact[usr][stime] = [etime, (etime-stime).total_seconds()//60, prev]
            print prev,"(",etime-stime+timedelta(seconds=10),")",
            stime = row['Time']
            gap = (stime-etime).total_seconds()
            #print usr, gap, etime, stime
            if (stime-etime)>=timedelta(days=1):
                print "Day end"
                print ""
            if gap>180:
                ugaps[usr].append(stime-etime)
                print "gap (",stime-etime+timedelta(seconds=10),")",
                act_seqs[usr].append(seq)
                seq = ''
            prev = row['Activity']
            seq += activity_encoding[prev]
            prev_time = stime
        elif (prev_time-row['Time']).total_seconds()>180:
            etime = prev_time
            uact[usr][stime] = [etime, (etime-stime).total_seconds()//60, prev]
            print prev,"(",etime-stime+timedelta(seconds=10),")",
            stime = row['Time']
            gap = (stime-etime)
            print "gap (",gap+timedelta(seconds=10),")",
            if (stime-etime)>=timedelta(days=1):
                print "Day end"
                print ""
            act_seqs[usr].append(seq)
            #print usr, gap, etime, stime
            seq = activity_encoding[prev]
            ugaps[usr].append(gap)
            prev_time = stime
        else:
            prev_time = row['Time']
            seq += activity_encoding[prev]
        '''
        getime = prev_time
    durations.append(getime-gstime)
    #print ":-:"
    #print act_seqs[usr]
    print "-:-"
    print "Total duration of recording:",(getime-gstime+timedelta(seconds=10))
#print uact
#print ugaps
df = pd.DataFrame(per_day_usr_activity, columns=['user','date','activity','start','end'])
print df.head()
pickle.dump(df, open('userwisedailyactivitytimeline.pickle', 'wb'))
'''
#for the activity sequences generated, find frequent patterns
pattern_lists = []
for usr in users:
    for seq in act_seqs[usr]:
        pattern_lists.append(seq)
#freq_seqs = seqmining.freq_seq_enum(pattern_lists, 4)
p = pyprefixspan(pattern_lists)
p.run()
p.setlen(3)
print "Frequent patterns:"
#print sorted(freq_seqs)
print p.out
print activity_encoding

pickle.dump(durations, open('durations.pickle','wb'))
pickle.dump(uact,open('userwise_activity.pickle', 'wb'))
pickle.dump(ugaps, open('userwise_gaps.pickle','wb'))
pickle.dump(act_seqs, open('userwiseactivity.pickle','wb'))

print "User durations:"
for i,dur in enumerate(durations):
    print users[i], dur
'''
