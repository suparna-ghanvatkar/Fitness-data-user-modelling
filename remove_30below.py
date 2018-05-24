'''
Script to remove segments less than 30 mins
'''
import pickle
import argparse
import pandas as pd
from datetime import timedelta, datetime,date

parser = argparse.ArgumentParser()
parser.add_argument('timeline',type=str,help='path to timeline file')
args = parser.parse_args()

df = pickle.load(open(args.timeline,'rb'))

df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])
users = df.user.unique()
for usr in users:
    udf = df.loc[df.user==usr]
    dates = udf.date.unique()
    for dat in dates:
        ddf = udf.loc[udf.date==dat]
        seqs = ddf.seq_no.unique()
        for seq in seqs:
            sdf = ddf.loc[ddf.seq_no==seq]
            sdf.sort_values(by='start')
            strt = sdf.iloc[0]['start']
            end = sdf.iloc[-1]['end']
            if (end-strt)<timedelta(minutes=30):
                df = df.loc[(df.user!=usr) | (df.date!=dat) | (df.seq_no!=seq)]
pickle.dump(df,open(args.timeline,'wb'))
