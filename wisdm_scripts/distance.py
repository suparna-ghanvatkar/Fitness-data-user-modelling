'''
Script takes two patterns and gives distance between them
'''
from fastdtw import fastdtw
import numpy as np
import math
from collections import Counter
from itertools import repeat
from scipy.spatial.distance import euclidean

def bring_to_len(pat, l):
    temp = []
    for p in pat:
        temp.extend(repeat(p,l))
    gran = len(temp)/l
    new_pat = []
    for i in range(len(temp)/gran):
        seq = temp[(i*gran):(i*gran)+gran]
        lab = Counter(seq).most_common(1)
        #print lab
        new_pat.append(lab[0][0])
    return new_pat

def distance(pat1, pat2):
    act_sed = {0:1,2:0,3:1,4:1,5:1,6:0}
    #strip of all -1
    pat1 = [p for p in pat1 if p!=-1]
    pat2 = [p for p in pat2 if p!=-1]
    count1 = dict(Counter(pat1))
#    count1 = {key:val/len(pat1) for key,val in count1.iteritems()}
    count2 = dict(Counter(pat2))
#    count2 = {key:val/len(pat2) for key,val in count2.iteritems()}
    dist_hist = math.sqrt(sum((count1[i]-count2[i])**2 for i in set(count1.keys()).intersection(set(count2.keys()))))
    spat1 = [act_sed[i] for i in pat1]
    spat2 = [act_sed[i] for i in pat2]
    count1 = Counter(spat1)
    count2 = Counter(spat2)
    active = int(bool(count1[1]) - bool(count2[1]))
    sdist_hist = math.sqrt(sum((count1[i]-count2[i])**2 for i in set(count1.keys()).intersection(set(count2.keys()))))
    if len(pat1)>len(pat2):
        d = len(pat1)*1.0/len(pat2)
        pat1 = bring_to_len(pat1, len(pat2))
    elif len(pat1)<len(pat2):
        d = len(pat2)*1.0/len(pat1)
        pat2 = bring_to_len(pat2, len(pat1))
    else:
        d = 0
    dist, path = fastdtw(pat1,pat2, dist=euclidean)
    spat1 = [act_sed[i] for i in pat1]
    spat2 = [act_sed[i] for i in pat2]
    #print spat1, spat2
    sdist,_ = fastdtw(spat1,spat2,dist=euclidean)
    #print dist, dist_hist, sdist, sdist_hist, d, active
    return math.sqrt(sum([dist**2, dist_hist**2, sdist**2, sdist_hist**2, d**2, active**2]))

