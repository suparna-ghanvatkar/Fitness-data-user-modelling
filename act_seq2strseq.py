# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:31:19 2018

@author: research
"""

import pickle

acts = pickle.load(open('activity_seq.dump','rb'))
seqs = []
for act_seq in acts:
    seq = ''
    for act in act_seq:
        seq += str(act)
    seqs.append(seq)
pickle.dump(seqs, open('activity_sequences.dump','wb'))