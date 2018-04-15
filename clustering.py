
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import os
import argparse
from sklearn.mixture import GaussianMixture
from sklearn import mixture
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('act', type=str, help='activity file path')
parser.add_argument('demo',type=str,help='demographics file path')
args = parser.parse_args()

#datadir = os.path.join('..', 'WISDM_at_v2.0')
#raw_headers = ['user','activity','time','x','y','z']
act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Activity','Time']
demo_headers = ['user','height','gender','age','weight','leg_injury']

act_data = pd.read_csv(args.act, header=None, names=act_headers)
demographics = pd.read_csv(args.demo, header=None, names=demo_headers)
act_data['Time'] = pd.to_datetime(act_data['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

'''
# In[14]:


labelled_act = os.path.join(datadir,'WISDM_at_v2.0_transformed.csv')
labelled_raw = os.path.join(datadir, 'WISDM_at_v2.0_raw.txt')
unlabelled_act = os.path.join(datadir,'WISDM_at_v2.0_unlabeled_transformed.csv')
unlabelled_raw = os.path.join(datadir, 'WISDM_at_v2.0_unlabeled_raw.txt')
unlabelled_raw_data = pd.read_csv(unlabelled_raw, header=None, names = raw_headers, parse_dates=True)
unlabelled_act_data = pd.read_csv(unlabelled_act, header=None, names = act_headers)
#raw_data = pd.read_csv(labelled_raw, header=None, names = raw_headers, parse_dates=True)
#act_data = pd.read_csv(labelled_act, header=None, names = act_headers)


# In[15]:


print "concat raw"
#raw_data = pd.concat([unlabelled_raw_data, raw_data])
raw_data = unlabelled_raw_data
act_data = unlabelled_act_data
#del unlabelled_raw_data
print "concat act"
#act_data = pd.concat([unlabelled_act_data, act_data])
#del unlabelled_act_data
print "concat over"

raw_data = raw_data.loc[raw_data['time']>0]
raw_data['time'] = pd.to_datetime(raw_data['time'], unit = 'ms', errors='coerce')


# In[16]:


users = list(set(raw_data.user.unique()).intersection(set(act_data.user.unique())))
print len(users)
time_user_raw = raw_data[['user','time']]
user_duration = time_user_raw.groupby('user').agg({'time':[min,max]})
user_duration.columns = ["_".join(x) for x in user_duration.columns.ravel()]
user_duration['duration'] = user_duration['time_max'] - user_duration['time_min']
#print user_duration.sort_values(by='duration').loc[act_data.user.unique()]['duration']
user_duration = user_duration.loc[users]
user_duration = user_duration[(user_duration['duration'].dt.total_seconds()//60.0) > 5]
print user_duration.sort_values(by='duration')['duration']
users = user_duration.index


# In[17]:

'''
users = act_data.user.unique()

activity_duration = {'Walking': 0, 'Jogging': 1, 'Standing': 2, 'Sitting': 3, 'Stairs': 4, 'LyingDown': 5}
user_feats = []
print users
nan_users = {}
for userid in users:
    user_act_data = act_data.loc[act_data['user']==userid][['Activity','Time']]
    print user_act_data.head()
    user_act_data = user_act_data.sort_values(by='Time')
    prev = user_act_data.iloc[0]['Activity']
    #feat = [walk_dur, jog_dur, stand_dur, sit_dur, stair_dur, ly_dur, no_tx, walk_jog, walk_sit, sit_walk]
    feat = [0.0]*7
    txfeat = [0.0]*36
    tot_dur = 0
    act_seq = []
    for i,row in user_act_data.iterrows():
        activity = row['Activity']
        if tot_dur<259200:
            if activity_duration.get(activity, -1)==-1:
                nan_users[userid] = i
                print userid, activity
            feat_index = activity_duration.get(activity, -1)
            #seq.append((user_no,tot_dur,feat_index))
            if feat_index != -1:
                act_seq.append(feat_index)
                feat[feat_index] += 1
                if activity!= prev:
                    feat[6] += 1
                    pfeat_index = activity_duration.get(activity)
                    index = (6*pfeat_index)+feat_index
                    txfeat[index] += 1
                prev = activity
            #else:
                #print userid, activity
            tot_dur = tot_dur + 1
        else:
            break
    dfeat = [x/tot_dur for x in feat[:-1] ]
    if feat[6]!=0:
        txfeat = [x/feat[6] for x in txfeat ]
    ufeat = dfeat+txfeat
    #act_feats.append(act_seq)
    user_feats.append(ufeat)
    #print act_seq
    #users.append(userid)
    #user_no = user_no + 1
    #users.append(userid)

import pickle
pickle.dump(user_feats, open('user_tx_feats.dump','wb'))


# In[13]:


#activity_duration.keys()


# In[6]:


#for userid in users:
#    user_act_data = act_data.loc[act_data['user']==userid]['Activity']
#    print user_act_data.head()


# In[18]:


print users
print len(users)
print len(user_feats)
ufeats = np.array(user_feats)
print ufeats


# In[10]:

'''
for userid in users:
    user_raw_data = raw_data.loc[raw_data['user']==userid]
    prev_time = user_raw_data.iloc[0]['time']
    gap = 0
    print userid, 'GAP LENGTHS'
    for index, attr in user_raw_data.iterrows():
        tgap = (attr[2] - prev_time).total_seconds()
        if  tgap > 120:
            gap = gap + 1
            print tgap//60, 'min..i.e.', tgap,'s'
        prev_time = attr[2]
    print '#gaps:',gap


# In[19]:


from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_validate

for n in range(2,10):
    gmm = GaussianMixture(n_components=n)
    cv_results = cross_validate(gmm, user_feats, cv=10,  return_train_score=False)
    print n, cv_results


# In[20]:


#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
n_components = range(2,16)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(ufeats) for n in n_components]
plt.plot(n_components, [m.bic(ufeats) for m in models], label='BIC')
plt.plot(n_components, [m.aic(ufeats) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')

'''
'''Temp comment
#Finding suitable GMM params

import matplotlib.pyplot as plt

n_components_range = range(2,16)
cv_types = ['spherical', 'tied', 'diag', 'full']
lowest_bic = np.infty
bic = []
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(ufeats)
        bic.append(gmm.bic(ufeats))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


models = [GaussianMixture(n, covariance_type='diag', random_state=0).fit(ufeats) for n in n_components]
plt.plot(n_components, [m.bic(ufeats) for m in models], label='BIC')
plt.plot(n_components, [m.aic(ufeats) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
# In[9]:
'''

from sklearn.cluster import DBSCAN
from sklearn import metrics
db = DBSCAN().fit(ufeats)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
db_labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(ufeats, db_labels))


# In[11]:


#cluster characteristics when 5 clusters for GMM and 3 for DBSCAN
gmm = GaussianMixture(5, covariance_type='diag', random_state=0).fit(ufeats)
user_labels_gmm = gmm.predict(ufeats)
print user_labels_gmm

def indices( mylist, value):
    return [i for i,x in enumerate(mylist) if x==value]

gmm_clusters = {}
for n in range(8):
   gmm_clusters[n] = indices(user_labels_gmm, n)

#dbscan clusters
db_clusters = {}
for n in range(2):
    db_clusters[n] = indices(db_labels, n)

print 'gmm:', gmm_clusters
print 'db:',db_clusters


# In[12]:


for cluster in gmm_clusters.keys():
    users_in_cluster = gmm_clusters[cluster]
    print 'CLUSTER ', cluster
    for u in users_in_cluster:
        print users[u],':',user_feats[u]

print 'DBSCAN'
for cluster in db_clusters.keys():
    users_in_cluster = db_clusters[cluster]
    print 'CLUSTER ', cluster
    for u in users_in_cluster:
        print users[u],':',user_feats[u]


# In[13]:


#len(raw_data.user.unique())


