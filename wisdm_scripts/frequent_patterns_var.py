'''
This script takes the activity pickle dataframe as input, encodes to character based activity sequence, and takes argument of granularity in seconds for freq pattern mining
Only sequences greater than 30 minutess considered
'''
import argparse
from collections import Counter
#from pyprefixspan import pyprefixspan
#from pymining import seqmining
#from prefixspan import PrefixSpan
import pickle
from collections import Counter
from itertools import product
from activity_plot import plot_timeline
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, date
import random

def is_subseq(seq, old_seq, start):
    if seq not in old_seq[0]:
        return False
    else:
        slen = len(seq)
        olen = len(old_seq[0])
        times = old_seq[2]
        for time in times:
            if start>=time and (start+timedelta(minutes=slen))<=(time+timedelta(minutes=olen)):
                return True
    return False

parser = argparse.ArgumentParser()
parser.add_argument('act',type=str,help='csv activity user file')
parser.add_argument('analysis',type=str,help='csv file to timeline analysis with users to be considered')
parser.add_argument('gran',type=int,help='seconds to consider as quantum')
args = parser.parse_args()

all_pats = []
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
act_legend = {'Standing':'0','gap':'1','Sitting':'2','Walking':'3','Stairs':'4','Jogging':'5','LyingDown':'6'}
#for i,a in enumerate(activities):
#    act_legend[str(i)] = a
legend = open(args.act+'_freq_legend.txt','w')
legend.write(str(act_legend))
legend.close()
for act,i in act_legend.iteritems():
    if act=='gap':
        gap_ind = i
#now change the legend by ignoring gaps
#act_legend = {x: act_legend[x] for x in act_legend if act_legend[x]!='gap'}
rand_nos = 0
tot_nos = 0
users = analysis.user.unique()
all_patterns = []
for usr in users:
    print("---------------------")
    print(usr)
    acts = []
    time_all_seq = []
    prop_counts = []
    user_data = act_data.loc[act_data.user==usr]
    dates = user_data.date.unique()
    for date in dates:
        date_wise = user_data.loc[user_data.date==date]
        date_wise = date_wise.sort_values(by='date')
        #if date_wise.iloc[-1]['end']-date_wise.iloc[0]['start'] > timedelta(minutes=30):
            #plot_data = date_wise[['activity','start','end']]
            #plot_data['start'] = [s.time() for s in plot_data['start']]
            #plot_data['end'] = [s.time() for s in plot_data['end']]
            #plot_timeline(plot_data,str(usr)+'_'+pd.to_datetime(date).strftime("Y-%m-%d")+'.svg')
        seqs = date_wise.seq_no.unique()
        for seq in seqs:
            print("new seq:")
            seq_str = ''
            act_seq = date_wise.loc[date_wise.seq_no==seq]
            act_seq = act_seq.sort_values(by='start')
            #print(act_seq[['activity','duration']])
            #start_time = datetime.combine(date.today(),act_seq.iloc[0]['start'])
            start_time = act_seq.iloc[0]['start']
            seq_str += act_legend[act_seq.iloc[0]['activity']]
            curr_time = start_time+quant
            row = act_seq.iloc[0]
            row_i = 0
            end = act_seq.iloc[-1]['end']
            if end-start_time >timedelta(minutes=30):
                while curr_time<end:
                    tot_nos += 1
                    if curr_time>row['end']:
                        #compare which part greater label that - majority vote
                        act_present = [act_legend[row['activity']]]
                        while curr_time>row['end']:
                            row_i += 1
                            row = act_seq.iloc[row_i]
                            act_present.append(act_legend[row['activity']])
                        acts_counts = Counter(act_present).most_common()
                        top_score = acts_counts[0][1]
                        top_acts = [a for a,val in acts_counts if val==top_score]
                        if len(top_acts)>1:
                            top_acts = [a for a in top_acts ]
                            lab = random.choice(top_acts)
                            rand_nos += 1
                        else:
                            lab = top_acts[0]
                        seq_str += str(lab)
                    else:
                        #still in this continue with this label
                        seq_str += act_legend[row['activity']]
                    curr_time += quant
                acts.append(seq_str)
                time_all_seq.append(start_time)
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
    seq_time = {}
    for k in range(25,4,-1):
        key_counts = Counter()
        tot_seqs = 0
        for i, seq in enumerate(acts):
            #find all substrings
            strt_time = time_all_seq[i]
            substrs = [(seq[i:i+k],strt_time+timedelta(minutes=i)) for i in range(len(seq)-k) if all(not is_subseq(seq[i:i+k],s,strt_time+timedelta(minutes=i)) for s in prop_counts)]
#            times = [strt_time+timedelta(minute=i) for i in range(len(seq)-k)]
            times = []
            temp = []
            for sub,time in substrs:
                temp.append(sub)
                times.append(time)
            substrs = temp
            tot_seqs += len(substrs)
            sub_counts = Counter(substrs)
            this_time = {}  #dict of occurence of each pf substring timing occ
            for s in sub_counts.keys():
                this_time[s] = []
                for i,q in enumerate(substrs):
                    if q==s:
                        this_time[s].append(times[i])
            key_counts.update(sub_counts)
            #also update the global timing info seq
            for key,val in this_time.iteritems():
                if key in seq_time.keys():
                    seq_time[key].extend(val)
                else:
                    seq_time[key] = val
        key_count = key_counts.most_common()
        for (key,val) in key_count:
            if (val*100)/tot_seqs :
                prop_counts.append([key,val*100.0/tot_seqs,seq_time[key], val, tot_seqs])
    prop_counts = sorted(prop_counts, key=lambda x: x[1], reverse=True)
    #chunk_size = len(prop_counts)/6
    #final_props = []
    #for i in range(6):
    #    patts = [x[0] for x in prop_counts[(i*chunk_size):(i*chunk_size)+chunk_size]]
        #fil_pats = []
    #    for i,p in enumerate(patts):
            #fil_pats.append(p)
            #print "eval for ",p
    #        if all([p not in s for s in [s for s in patts if s!=p]]):
    #            final_props.append(prop_counts[i])
    for seq, prop, time, num, deno in prop_counts:
        times = [time[0]]
        for t in time:
            if (t not in times) and (all(t+timedelta(minutes=i) not in times for i in range(5))) and all(t-timedelta(minutes=i) not in times for i in range(5)):
                times.append(t)
                print type(t)
        times = [datetime.time(ti).strftime("%H:%M:%S") for ti in times]
        text = str(seq)+' '+str(prop)+' '+str(times)+' '+str(num)+' '+str(deno)+'\n'
        f.write(text)
    f.close()
    #for pair in final_props[:30]:
    #    all_pats.append(pair[0])
    #pickle.dump(all_pats,open('wisdm_patts.pickle','wb'))
    '''
    p = pyprefixspan(acts)
    p.run()
    p.setlen(4)
    print usr, p.out
    frq_seqs = seqmining.freq_seq_enum(acts, 4)
    print(sorted(frq_seqs))
    ps = PrefixSpan(acts)
    #print(ps.frequent(3))
    ps.minlen=4
    ps.maxlen=4
    freq_pats = ps.topk(10)
    for matches,patt in freq_pats:
        all_patterns.append(patt)
pickle.dump(all_patterns, open('wisdm_freq_patts.pickle','wb'), 2)
    '''
print rand_nos*1.0/tot_nos

