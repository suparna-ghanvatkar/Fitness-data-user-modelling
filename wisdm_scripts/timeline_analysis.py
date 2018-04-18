'''
Script to perform analysis on user activity timeline given as argument
'''
import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument('timeline',type=str,help='path to timeline file')
parser.add_argument('gaps',type=str,help='path to gaps file')
parser.add_argument('outpath',type=str,help='path to output csv analysis file')
args = parser.parse_args()

dateparse = lambda x: pd.datetime.strptime(x,"%Y-%m-%d")
duration_parse = lambda x: pd.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
timeparse = lambda x: pd.datetime.strptime(x,"%H:%M:%S")
timeline = pd.read_csv(args.timeline, parse_dates = ['date','start','end'])
gaps = pd.read_csv(args.gaps, parse_dates = ['date','start','end'])
gaps['duration'] = pd.to_timedelta(gaps['duration'])
timeline['duration'] = pd.to_timedelta(timeline['duration'])
print timeline.head()
print gaps.head()

#print timeline.dtypes
#print gaps.dtypes
activities = timeline.activity.unique()
analysis = []
users = timeline.user.unique()
for usr in users:
    user_data = timeline.loc[timeline.user==usr]
    dates = user_data.date.unique()
    long_seq_data = user_data
    longest_seq = timedelta(0)
    no_seqs = 0
    for date in dates:
        day_act = user_data.loc[user_data.date==date]
        seqs_in_day = day_act.seq_no.unique()
        for seq in seqs_in_day:
            seq_data = day_act.loc[day_act.seq_no==seq]
            no_seqs += 1
            if (seq_data.iloc[-1]['end']-seq_data.iloc[0]['start'])>longest_seq:
                longest_seq = seq_data.iloc[-1]['end']-seq_data.iloc[0]['start']
                long_seq_data = seq_data
    act_dur = [timedelta(0)]*len(activities)
    #print long_seq_data.head()
    for i,act in enumerate(activities):
        act_data = long_seq_data.loc[long_seq_data.activity==act]['duration']
        for _,row in act_data.iteritems():
            act_dur[i] += row
    tot_mins = [a.total_seconds()/60 for a in act_dur]
    tot_dur = sum(tot_mins)
    prop_mins = [a*1.0/tot_dur for a in tot_mins]
    user_gaps = gaps.loc[gaps.user==usr]
    #print user_gaps['duration'].describe()
    max_gap = np.max(user_gaps.duration.dt.total_seconds())/60
    mean_gap = np.mean(user_gaps.duration.dt.total_seconds())/60
    analysis.append([usr, len(dates), no_seqs, longest_seq, max_gap,mean_gap]+act_dur+prop_mins)

cols = ['user','no_days', 'no_cont_segs','longest seg','max gap time','mean gap time']+list(activities)+list(activities)
print cols
df = pd.DataFrame(analysis, columns=cols)
df = df.sort_values(by='longest seg', ascending=False)
print df.head()
df.to_csv(args.outpath,index=False)
