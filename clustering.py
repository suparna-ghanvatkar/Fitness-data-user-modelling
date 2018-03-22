
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import os

datadir = os.path.join('..', 'WISDM_at_v2.0')
raw_headers = ['user','activity','time','x','y','z']
act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Activity']


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


activity_duration = {'Walking': 0, 'Jogging': 1, 'Standing': 2, 'Sitting': 3, 'Stairs': 4, 'LyingDown': 5}
user_feats = []
#users = []
nan_users = {}
for userid in users:
    user_act_data = act_data.loc[act_data['user']==userid]['Activity']
    prev = user_act_data.iloc[0]
    #feat = [walk_dur, jog_dur, stand_dur, sit_dur, stair_dur, ly_dur, no_tx, walk_jog, walk_sit, sit_walk]
    feat = [0.0]*10
    tot_dur = 0
    for i,activity in user_act_data.iteritems():
        
        if activity_duration.get(activity, -1)==-1:
            nan_users[userid] = i        
            print userid, activity
        feat_index = activity_duration.get(activity, -1)
        if feat_index != -1:
            #print activity
            feat[feat_index] = feat[feat_index] + 1
            if activity!= prev:
                feat[6] = feat[6] + 1
                if prev=='Walking' and activity=='Jogging':
                    feat[7] = feat[7] + 1
                elif prev=='Walking' and activity=='Sitting':
                    feat[8] = feat[8] + 1
                elif prev=='Sitting' and activity=='Walking':
                    feat[9] = feat[9] + 1
            prev = activity
        #else:
            #print userid, activity
        tot_dur = tot_dur + 1
    ufeat = [x/tot_dur for x in feat]
    user_feats.append(ufeat)
    #users.append(userid)


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
'''

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


# In[9]:


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
gmm = GaussianMixture(8, covariance_type='full', random_state=0).fit(ufeats)
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


len(raw_data.user.unique())


