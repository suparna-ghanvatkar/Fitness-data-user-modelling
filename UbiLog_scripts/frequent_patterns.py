'''
This script takes the activity pickle dataframe as input, encodes to character based activity sequence, and takes argument of granularity in seconds for freq pattern mining
Only sequences greater than 30 minutes considered
'''
import argparse
from collections import Counter
#from pyprefixspan import pyprefixspan
#from pymining import seqmining
#from prefixspan import PrefixSpan
from itertools import product
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, date, time

parser = argparse.ArgumentParser()
parser.add_argument('act',type=str,help='csv activity user file')
parser.add_argument('analysis',type=str,help='csv file to timeline analysis with users to be considered')
parser.add_argument('gran',type=int,help='seconds to consider as quantum')
args = parser.parse_args()

analysis = pd.read_csv(args.analysis)
#act_data = pickle.load(open(args.act,'rb'))
act_data = pd.read_csv(args.act, parse_dates=['date','start','end'])
act_data['duration'] = pd.to_timedelta(act_data['duration'])

quant = timedelta(seconds = args.gran)
activities = list(act_data.activity.unique())
#act_data['start'] = [datetime.combine(date.today(),a) for _,a in act_data['start'].iteritems()]
#act_data['end'] = [datetime.combine(date.today(),a) for _,a in act_data['end'].iteritems()]
act_data['start'] = act_data['start'].dt.to_pydatetime()
act_data['end'] = act_data['end'].dt.to_pydatetime()
act_legend = {}
for i,a in enumerate(activities):
    act_legend[str(i)] = a
legend = open(args.act+'_freq_legend.txt','wb')
print>>legend, act_legend
users = analysis.user.unique()
for usr in users:
    print("---------------------")
    print(usr)
    acts = []
    user_data = act_data.loc[act_data.user==usr]
    dates = user_data.date.unique()
    for date in dates:
        date_wise = user_data.loc[user_data.date==date]
        seqs = date_wise.seq_no.unique()
        for seq in seqs:
            print("new seq:")
            seq_str = ''
            act_seq = date_wise.loc[date_wise.seq_no==seq]
            act_seq = act_seq.sort_values(by='start')
            print(act_seq[['activity','duration']])
            #start_time = datetime.combine(date.today(),act_seq.iloc[0]['start'])
            start_time = act_seq.iloc[0]['start']
            seq_str += str(activities.index(act_seq.iloc[0]['activity']))
            curr_time = start_time+quant
            row = act_seq.iloc[0]
            row_i = 0
            end = act_seq.iloc[-1]['end']
            if end-start_time >timedelta(minutes=30):
                if start_time.time()<time(hour =7):
                    if end.time()<time(hour=7):
                        continue
                    while row['end'].time()<time(hour=7):
                        print row
                        row_i += 1
                        row = act_seq.iloc[row_i]
                    start_time = time(hour=7)
                while curr_time<end:
                    if curr_time>row['end']:
                        #compare which part greater label that - majority vote
                        act_present = [activities.index(act_seq.iloc[0]['activity'])]
                        while curr_time>row['end']:
                            row_i += 1
                            row = act_seq.iloc[row_i]
                            act_present.append(activities.index(act_seq.iloc[0]['activity']))
                        lab,_ = Counter(act_present).most_common(1)[0]
                        seq_str += str(lab)
                    else:
                        #still in this continue with this label
                        seq_str += str(activities.index(row['activity']))
                    curr_time += quant
                acts.append(seq_str)
    #print usr,":"
    #print acts
    f = open(str(usr)+'_act_seq.txt','w')
    for act in acts:
        for item in act:
            f.write("%s " % item)
        f.write("\n")
    f.close()
    #now check for count of substrings of given length:
    f = open(str(usr)+'_freq_pats.txt','w')
    keywords = [''.join(i) for i in product(''.join(act_legend), repeat = args.len)]
    key_counts = dict.fromkeys(keywords,0)
    print key_counts
    for key in keywords:
        for seq in acts:
            key_counts[key] += seq.count(key)
    key_count = key_counts.items()
    key_count.sort(key=lambda x: x[1], reverse=True)
    print key_counts, key_count
    for pair in key_count:
        print>>f, pair
    f.close()
    '''
    p = pyprefixspan(acts)
    p.run()
    p.setlen(4)
    print usr, p.out
    frq_seqs = seqmining.freq_seq_enum(acts, 4)
    print(sorted(frq_seqs))
    ps = PrefixSpan(acts)
    print(ps.frequent(3))
    print(ps.topk(5))
    '''
