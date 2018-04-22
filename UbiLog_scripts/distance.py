'''
Script takes two patterns and gives distance between them
'''
from fastdtw import fastdtw
import numpy as np
import math
from collections import Counter
from scipy.spatial.distance import euclidean

act_to_sed = {'0':'0','1':'0','3':'0','4':'1','5':'1'}
 def distance(pat1, pat2):
    dist, path = fastdtw(list(pat1),list(pat2), dist=euclidean)
    count1 = Counter(pat1)
    count2 = Counter(pat2)
    dist_hist = math.sqrt(sum(count1[i]-count2[i])**2 for i in act_sed.keys())
    spat1 = [act_to_sed[i] for i in pat1]
    spat2 = [act_to_sed[i] for i in pat2]
    sdist,_ = fastdtw(spat1,spat2,dist=euclidean)
    count1 = Counter(spat1)
    count2 = Counter(spat2)
    sdist_hist = math.sqrt(sum(count1[i]-count2[i])**2 for i in act_sed.keys())


