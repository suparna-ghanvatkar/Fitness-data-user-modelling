'''
Script to perform clustering on all patterns using distance metric. Argument pickle file
'''
import pickle
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from distance import distance

pats = pickle.load(open('ubiqlog_patts.pickle','rb'))
max_len = 0
for p in pats:
    if len(p)>max_len:
        max_len = len(p)
num_pats = []
for p in pats:
    pat = []
    for s in p:
        pat.append(int(s))
    while len(pat)<max_len:
        pat.append(-1)
    num_pats.append(pat)
feats = np.array(num_pats)
#print feats
labels = fclusterdata(feats, 5, criterion='maxclust', metric=distance)
print labels
print pats
