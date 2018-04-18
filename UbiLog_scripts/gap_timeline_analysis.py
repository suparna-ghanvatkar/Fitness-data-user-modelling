'''
This script performs analysis of gaps and generates timeline for user

'''

import argparse
import pandas as pd
import pickle
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument('data',type=str,help='path to data file')
args = parser.parse_args()

per_day_usr_activity = []
gaps = []
dateparse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
dataset = pd.read_csv(args.data, parse_dates=['Start','End'], date_parser=dateparse)
users = dataset.User.unique()
for usr in users:
    user_act = dataset.loc[dataset.User==usr]
    dates = user_act.Date.unique()
    for date in dates:
        print "Date:", date
        date_act = user_act.loc[user_act.Date==date]
        date_act = date_act.sort_values(by='Start')
        stime = date_act.iloc[0]['Start']
        gstime = stime
        prev_time = date_act.iloc[0]['End']
        prev_act = date_act.iloc[0]['Activity']
        for i,row in date_act.iterrows():
            curr_time = row['Start']
            curr_act = row['Activity']
            if curr_time-prev_time>timedelta(seconds=20):
                print prev_act, prev_time-stime," ",stime," ",prev_time
                per_day_usr_activity.append([usr,date,prev_act,prev_time-stime,stime,prev_time])
                print "gap",curr_time-prev_time," ",prev_time," ",curr_time
                gaps.append([usr,date,curr_time-prev_time,prev_time,curr_time])
                stime = curr_time
            elif curr_act!=prev_act:
                print prev_act, prev_time-stime," ",stime," ",prev_time
                per_day_usr_activity.append([usr,date,prev_act,prev_time-stime,stime,prev_time])
                stime = curr_time
            prev_time = row['End']
            prev_act = curr_act
            getime = prev_time
        print"-------------------------"
        print "Duration of recording in day:",getime-gstime, gstime,getime
        print":::::::::::::::::::::::::::::::"
        print "------------------------------"

df = pd.DataFrame(per_day_usr_activity, columns = ['user','date','activity','duration','start','end'])
gp = pd.DataFrame(gaps, columns=['user','date','duration','start','end'])

dates_per_user = []
#Printing gap statistics:
for usr in users:
    usr_gap = gp.loc[gp.user==usr]
    dates = gp.date.unique()
    dates_per_user.append(len(dates))
    for date in dates:
        date_gap = usr_gap.loc[usr_gap.date==date][['duration','start','end']]
        print date
        print date_gap.describe()

print "No of days of recordings:",dates_per_user, " = ",sum(dates_per_user)

pickle.dump(df,open('userwiseactivitytimeline.pickle','wb'))
pickle.dump(gp,open('userwisegaps.pickle','wb'))
