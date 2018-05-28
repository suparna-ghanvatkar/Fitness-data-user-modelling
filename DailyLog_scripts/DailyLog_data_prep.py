'''
Script to concat all activity data and then create activity timeline for all users for further ananlysis
'''

import argparse
import json
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument('data_dir',type=str, help='path to folder containing extracted files of all subjects acc data')
parser.add_argument('data_path',type=str, help='path to output csv')
args = parser.parse_args()

act_data = pd.DataFrame()
dateparse = lambda x: pd.datetime.strptime(x, "%d-%m-%y %H:%M:%S")

usr_ids = [o for o in os.listdir(args.data_dir)]
for u in usr_ids:
    udata = pd.DataFrame()
    udir = os.path.join(args.data_dir, o)
    files = [os.path.join(udir,f) for f in os.listdir(udir)]
    for f in files:
        day_dat = pd.read_csv(f)
        day_dat['attr_time'] = day_dat['attr_time'].dt.to_pydatetime()
        day_data = day_dat['attr_time','label_posture']

        udata = pd.concat([udata,day_dat])
    udata['user'] = int(u)
    act_data = pd.concat([act_data,udata])

'''
Act data has user, time and label posture..create this into start end activity user type of data
'''
activity_data = []
act_data = act_data[act_data.label_posture!='unknown']
first_row = act_data[0]
prev_usr = first_row['user']
prev_date = first_row['attr_time'].date()
prev_start = first_row['attr_time'].time()
prev_time = prev_start
prev_act = first_row['label_posture']
for index,row in act_data.iterrows():
    user = row['user']
    date = row['attr_time'].date()
    time = row['attr_time'].time()
    act = row['label_posture']
    if user==prev_usr and date==prev_date and act==prev_act and time-prev_time<timedelta(minutes=1):
        prev_time = time
    else:
        prev_end = prev_time
        activity_data.append(prev_usr, prev_date, prev_act, prev_start, prev_end)
        prev_usr = user
        prev_date = date
        prev_start = time
        prev_time = time
        prev_act = act
prev_end = prev_time
activity_data.append(prev_usr, prev_date, prev_act, prev_start, prev_end)
Act_Data = pd.DataFrame(activity_data, columns = ['User','Date','Activity','Start','End'])
print Act_Data.head()
Act_Data.to_csv(args.data_path,index=False)