'''
This script is created to preprocess the data from UbiLog dataset

'''
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('records',type=str, help='path to file with lines filtered regarding activity')
parser.add_argument('data_path',type=str, help='path to output csv')
args = parser.parse_args()

act_data = []

data = open(args.records,'rb')
for line in data:
    contents = line.split(':',1)
    act = contents[1]
    content = contents[0].split('/')
    user_id,gender = content[1].split('_')
    rec = json.JSONDecoder().decode(act)
    act_log = rec['Activity']
    act_log['start'] = datetime.strptime(act_log['start'], '%m-%d-%Y %H:%M:%S')
    act_log['end'] = datetime.strptime(act_log['end'], '%m-%d-%Y %H:%M:%S')
    curr_date = act_log['start'].date()
    act_log['condfidence'] = float(act_log['condfidence'])
    #print act_log
    act_data.append([user_id, gender, curr_date, act_log['type'], act_log['start'], act_log['end'], act_log['condfidence']])

Act_Data = pd.DataFrame(act_data, columns = ['User','Gender','Date','Activity','Start','End','Confidence'])
print Act_Data.head()
Act_Data.to_csv(args.data_path,index=False)
