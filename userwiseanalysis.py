import pickle
import os
import pandas as pd

lab_raw = pickle.load(open('filtered_raw_lab.csv','rb'))
lab_act = pickle.load(open('filtered_act_lab.csv','rb'))
unlab_raw = pickle.load(open('filtered_raw_unlab.csv','rb'))
unlab_act = pickle.load(open('filtered_act_unlab.csv','rb'))
demo = pickle.load(open('filtered_demo.csv','rb'))

users = demo.user.unique()
for user in users:
    raw_rows = lab_raw.loc[lab_raw['user']==user].shape[0]
    act_rows = lab_act.loc[lab_act['user']==user].shape[0]
    if raw_rows!=(200*act_rows):
        print user,"less labelled rows"
    raw_rows = unlab_raw.loc[unlab_raw['user']==user].shape[0]
    act_rows = unlab_act.loc[unlab_act['user']==user].shape[0]
    if raw_rows!=(200*act_rows):
        print user,"less unlabelled rows", str(raw_rows), str(act_rows)

#user activtiy timeline
uact = {}
ugaps = {}
for usr in users:
    uact[usr] = {}
    ugaps[usr] = []
for userid in users:
    gap_stats = []
    maxgap = 0
    mingap = 0
    sumgap = 0
    user_raw_data = lab_raw.loc[lab_raw['user']==userid]
    user_raw_data = user_raw_data.sort_values('time')
    if not user_raw_data.empty:
        stime = user_raw_data.iloc[0]['time']
        etime = stime
        prev = user_raw_data.iloc[0]['activity']
        gstime = stime
        for i,attr in user_raw_data.iterrows():
            if attr[1]!=prev:
                #print attr[1], prev
                #raw_input()
                uact[userid][stime] = [etime, (etime-stime).total_seconds()//60, prev]
                stime = attr[2]
                gap = (stime-etime).total_seconds()
                print usr, gap, etime, stime
                if gap>120:
                    ugaps[usr].append(gap)
                etime = stime
                prev = attr[1]
            else:
                etime = attr[2]
            getime = etime
    print userid, ( getime - gstime ).total_seconds()
'''
    user_raw_data = unlab_raw.loc[unlab_raw['user']==userid]
    user_raw_data = user_raw_data.iloc[::200,:]
    user_act_data = unlab_act.loc[unlab_act['user']==userid]
    if not user_act_data.empty:
        stime = user_raw_data.iloc[0]['time']
        etime = stime
        prev = user_act_data.iloc[0]['Activity']
        for i,attr in user_act_data.iterrows():
            if attr[44]!=prev:
                uact[userid][stime] = [etime, (etime-stime).total_seconds()//60, prev]
                stime = user_raw_data.iloc[i]['time']
                gap = (stime-etime).total_seconds()
                if gap>120:
                    ugaps[usr].append(gap)
                etime = stime
            else:
                etime = user_raw_data.iloc[i]['time']
'''
