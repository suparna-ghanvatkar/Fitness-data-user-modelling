'''
Script to perform analysis on user activity timeline given as argument
'''
import pickle
import argparse
import pandas as pd
import numpy as np
from datetime import timedelta,datetime,time
from activity_plot import plot_timeline

parser = argparse.ArgumentParser()
parser.add_argument('timeline',type=str,help='path to timeline pickle file')
parser.add_argument('gaps',type=str,help='path to gaps pickle file')
parser.add_argument('outpath',type=str,help='path to output csv analysis file')
args = parser.parse_args()

dateparse = lambda x: pd.datetime.strptime(x,"%Y-%m-%d")
duration_parse = lambda x: pd.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
timeparse = lambda x: pd.datetime.strptime(x,"%H:%M:%S")
#timeline = pd.read_csv(args.timeline, parse_dates = ['date','start','end'])
#gaps = pd.read_csv(args.gaps, parse_dates = ['date','start','end'])
#gaps['duration'] = pd.to_timedelta(gaps['duration'])
#timeline['duration'] = pd.to_timedelta(timeline['duration'])
#timeline['start'] = pd.to_datetime(timeline['start'],format="%H:%M:%S")
#timeline['end'] = pd.to_datetime(timeline['end'],format="%H:%M:%S")
timeline = pickle.load(open(args.timeline,'rb'))
gaps = pickle.load(open(args.gaps,'rb'))
#print timeline.head()
#print gaps.head()

print timeline.dtypes
#print gaps.dtypes
activities = timeline.activity.unique()
analysis = []
#temp only for selected users...plot all days
#df = pd.read_csv('wisdm_all_4hrs_users_5mins.csv')
#users = df.user.unique()
users = timeline.user.unique()
for usr in users:
    user_data = timeline.loc[timeline.user==usr]
    dates = user_data.date.unique()
    long_seq_data = user_data
    longest_seq = timedelta(0)
    no_seqs = 0
    #print usr
    seq_durs = []
    for date in dates:
        day_act = user_data.loc[user_data.date==date]
        seqs_in_day = day_act.seq_no.unique()
        #print date
        for seq in seqs_in_day:
            seq_data = day_act.loc[day_act.seq_no==seq]
            no_seqs += 1
            seq_strt = datetime.combine(date.today(), seq_data.iloc[0]['start'])
            seq_end = datetime.combine(date.today(), seq_data.iloc[-1]['end'])
            seq_durs.append((seq_end-seq_strt).total_seconds())
            #for ubiq data uncommment next two lines
            #if (seq_data.iloc[-1]['end'].to_pydatetime()-seq_data.iloc[0]['start'].to_pydatetime())>longest_seq:
            #    longest_seq = seq_data.iloc[-1]['end'].to_pydatetime()-seq_data.iloc[0]['start'].to_pydatetime()
            if (datetime.combine(date.today(),seq_data.iloc[-1]['end'])-datetime.combine(date.today(),seq_data.iloc[0]['start']))>longest_seq:
                longest_seq = datetime.combine(date.today(),seq_data.iloc[-1]['end'])-datetime.combine(date.today(),seq_data.iloc[0]['start'])
            #print seq_data.iloc[-1]
                #print seq_data.tail(1)
                #print seq_data.iloc[0]
                long_seq_data = seq_data
        try:
            plot_timeline(day_act, str(usr)+'_'+str(date)+'.svg')
        except Exception as e:
            print usr,date,e
    #find mean and std of all durations:
    mean_dur = datetime.utcfromtimestamp(np.mean(seq_durs)).strftime("%H:%M:%S")
    std_dur = datetime.utcfromtimestamp(np.std(seq_durs)).strftime("%H:%M:%S")
    act_dur = [timedelta(0)]*len(activities)
    tot_act_dur = [timedelta(0)]*len(activities)
    #print long_seq_data.head()
    #print long_seq_data.iloc[0]
    #print long_seq_data.iloc[-1]
    plot_data = long_seq_data[['activity','start','end']]
    #print plot_data.dtypes
    #plot_data['start'] = [s.time() for s in plot_data['start']]
    #plot_data['end'] = [e.time() for e in plot_data['end']]
    #plot_timeline(plot_data, str(usr)+'longest.svg')
    for i,act in enumerate(activities):
        tot_act = user_data.loc[user_data.activity==act]['duration']
        act_data = long_seq_data.loc[long_seq_data.activity==act]['duration']
        for _,row in act_data.iteritems():
            act_dur[i] += row
        for _,row in tot_act.iteritems():
            tot_act_dur[i] += row
    long_seq_data = long_seq_data.sort_values(by='start')
    print "Printing Longest sequence for user:",usr
    for i,row in long_seq_data.iterrows():
        print row['activity'],"(",row['duration'],")",
    print ""
    tot_mins = [a.total_seconds()/60 for a in act_dur]
    tot_dur = sum(tot_mins)
    prop_mins = [int((a*1.0/tot_dur)*100) for a in tot_mins]
    tot_mins = [ datetime.utcfromtimestamp(a.total_seconds()).strftime("%H:%M:%S") for a in act_dur]
    tot_all_mins = [a.total_seconds()/60 for a in tot_act_dur]
    tot_all_dur = sum(tot_all_mins)
    prop_all_mins = [int((a*1.0/tot_all_dur)*100) for a in tot_all_mins]
    user_gaps = gaps.loc[gaps.user==usr]
    if user_gaps.empty:
        max_gap=0
        mean_gap=0
    #print user_gaps['duration'].describe()
    else:
        max_gap = np.max(user_gaps.duration.dt.total_seconds())
        mean_gap = np.mean(user_gaps.duration.dt.total_seconds())
    #if longest_seq>timedelta(hours=4):
    analysis.append([usr, len(dates), no_seqs, (datetime(1970,1,1)+timedelta(seconds=longest_seq.total_seconds())).strftime("%H:%M:%S"), (datetime(1970,1,1)+timedelta(seconds=max_gap)).strftime("%H:%M:%S"), (datetime(1970,1,1)+timedelta(mean_gap)).strftime("%H:%M:%S")]+tot_mins+prop_mins+tot_act_dur+[ tot_all_dur, mean_dur, std_dur ]+prop_all_mins)

cols = ['user','no_days', 'no_seqs','longest seg','max gap time','mean gap time']+list(activities)+list(activities)+list(activities)+['total_duration', 'mean seg len', 'std dev of seg len']+list(activities)
#print cols
df = pd.DataFrame(analysis, columns=cols)
df = df.sort_values(by='longest seg', ascending=False)
#print df.head()
df.to_csv(args.outpath,index=False)
#now find aggregate stats:
#print "#users:", len(df)
#print "mean of no of days recorded:", df['no_days'].mean()
#print "std dev of no of days of recording:", df['no_days'].std()
#df['longest seq'] = pd.to_datetime(df['longest seq'])
#df['max gap time'] = pd.to_datetime(df['max_gap_time'])
