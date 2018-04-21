
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import os
import argparse
from sklearn.mixture import GaussianMixture
from sklearn import mixture
import pickle
import itertools

parser = argparse.ArgumentParser()
#parser.add_argument('act', type=str, help='activity file path')
#parser.add_argument('demo',type=str,help='demographics file path')
parser.add_argument('patterns',type=str,help = 'path to pickle file with frequent pattenrs')
args = parser.parse_args()

#datadir = os.path.join('..', 'WISDM_at_v2.0')
#raw_headers = ['user','activity','time','x','y','z']
act_headers = ['user','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','Resultant','Activity','Time']
demo_headers = ['user','height','gender','age','weight','leg_injury']

#act_data = pd.read_csv(args.act, header=None, names=act_headers)
#demographics = pd.read_csv(args.demo, header=None, names=demo_headers)
#act_data['Time'] = pd.to_datetime(act_data['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
ufeats = np.array(pickle.load(open(args.patterns,'rb')))


#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
n_components = range(2,16)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(ufeats) for n in n_components]
plt.plot(n_components, [m.bic(ufeats) for m in models], label='BIC')
plt.plot(n_components, [m.aic(ufeats) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')

#Finding suitable GMM params

#import matplotlib.pyplot as plt

n_components_range = range(2,16)
cv_types = ['spherical', 'tied', 'diag', 'full']
lowest_bic = np.infty
bic = []
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
        gmm.fit(ufeats)
        bic.append(gmm.bic(ufeats))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
print bic
print best_gmm
#bic = np.array(bic)
#color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
#                              'darkorange'])
#clf = best_gmm
#bars = []
'''
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
'''
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
gmm = GaussianMixture(4, covariance_type='full', random_state=0).fit(ufeats)
#gmm = best_gmm
user_labels_gmm = gmm.predict(ufeats)
print user_labels_gmm

def indices( mylist, value):
    return [i for i,x in enumerate(mylist) if x==value]

gmm_clusters = {}
for n in range(10):
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
        print ufeats[u]
print 'DBSCAN'
for cluster in db_clusters.keys():
    users_in_cluster = db_clusters[cluster]
    print 'CLUSTER ', cluster
    for u in users_in_cluster:
        print ufeats[u]

# In[13]:

print len(ufeats), ufeats
#len(raw_data.user.unique())


