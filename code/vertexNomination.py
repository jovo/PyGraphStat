'''
Created on Mar 22, 2012

@author: dsussman
'''

from itertools import permutations
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np
import random

from rpy2 import robjects
from rpy2.robjects import numpy2ri

def num_diff_w_perms(l1, l2):
    """Compute min_{p in perm} |p(l1)-l2|_1
    
    This loops over all permutations so could be slow for many groups""" 
    label = list(set(l1))
    
    min_diff = np.Inf
    for p in permutations(label):
        l1p = [p[label.index(l)] for l in l1]
        min_diff = min(min_diff,metrics.zero_one(l1p, l2))
    return min_diff


def confusion_matrix(g1, g2, plot_table=True):
    assert(len(g1)==len(g2))
    n = len(g1)
    s1 = list(set(g1))
    s2 = list(set(g2))
    n1 = len(s1)
    n2 = len(s2)

    cm = np.array([[len([v for v in np.arange(n) if g1[v]==l1 and g2[v]==l2]) 
           for l1 in s1] for l2 in s2])
    norm_cm = (np.diag(1.0/cm.dot(np.ones(cm.shape[1])))).dot(cm)
    
    cm_total = np.zeros((n2+1,n1+1))
    cm_total[0:n2,0:n1] = cm
    cm_total[0:n2,n1] = cm.dot(np.ones(n1))
    cm_total[n2,0:n1] = np.ones(n2).dot(cm)
    cm_total[n2,n1] = np.sum(cm_total[0:n2,n1])
    
    if plot_table:
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.matshow(norm_cm)
        
        plt.xlim(-.5, n1+.5)
        plt.ylim(n2+.5, -.5)
        
        for x in xrange(n1+1):
            for y in xrange(n2+1):
                ax.annotate(str(int(cm_total[y][x])), xy=(x, y), 
                            horizontalalignment='center',
                            verticalalignment='center')
        
        cb = fig.colorbar(res)
        s1.append('Total')
        s2.append('Total')
        plt.xticks(range(n1+1), s1, rotation=45)
        plt.yticks(range(n2+1), s2)
        
        
    return cm

class mclust_performance(object):
    mclustRes = None
    labels = []
    shape = None
    
    _khat = None
    _Lhat = None
    _ari = None
    
    def __init__(self,labels, x=None):
        self.labels = labels
        if x is not None:
            self.run(x)
    
    def run(self, x, init=None, kRange = None):
        self.shape = x.shape
        numpy2ri.activate()
        r = robjects.r
        r.options(warn=-1)
        r.library('mclust')
        
        if kRange is None:
            kRane = arange(1,10)
        
        if init is None:
            mcr = r.Mclust(x)
        else:
            subset = random.sample(np.arange(self.shape[0]),init)
            subsetR = r['list'](subset=subset)
            mcr = r.Mclust(x,initialization=subsetR)
        self.mclustRes = dict([(i[0],i[1]) for i in mcr.iteritems()])
        return self
    
    def get_khat(self):
        if self.mclustRes is None:
            print "No results yet. Use run(x)"
        if self._khat is None:
            self._khat = self.mclustRes['G']
        return self._khat
    
    def get_lhat(self):
        if self.mclustRes is None:
            print "No results yet. Use run(x)"
        if self._Lhat is None:
            mc_class = np.array(self.mclustRes['classification']).astype(int)-1
            self._Lhat = num_diff_w_perms(mc_class,self.labels)/self.shape[0]
        return self._Lhat
    
    def get_ari(self):
        if self.mclustRes is None:
            print "No results yet. Use run(x)"
        if self._ari is None:
            mc_class = np.array(self.mclustRes['classification']).astype(int)-1
            self._ari = metrics.adjusted_rand_score(mc_class,self.labels)
        return self._ari

class kmeans_performance(object):
    x = None
    n = 0
    true_labels = []
    
    max_log_nSquareErr = 3.0/8.0+.0001
    kRange = []
    
    ari = None
    L = None
    squaredError = None
    
    kHat = 0
    
    
    def __init__(self,x, labels, kRange, max_log_nSquareErr=3.0/8.0+.0001):
        self.x = x
        self.n = x.shape[0]
        self.true_labels = labels
        self.kRange = kRange
        self.max_log_nSquareErr = max_log_nSquareErr
        
    def run(self, clear=True):
        nK = len(self.kRange)
        self.ari = np.zeros(nK)
        self.L = np.zeros(nK)
        self.squaredError= np.zeros(nK)
        for i in xrange(nK):
            k = self.kRange[i]
            km = KMeans(k=k, init='k-means++', n_init=3,
                        max_iter=300,copy_x=False)
            km.fit(self.x)
            self.ari[i] = metrics.adjusted_rand_score(self.true_labels,km.labels_)
            self.L[i] = num_diff_w_perms(self.true_labels,km.labels_)
            self.squaredError[i]= km.inertia_
            
        goodk=[self.kRange[i] for i in xrange(nK) 
              if np.log(self.squaredError[i])/(2*np.log(self.n))
                <self.max_log_nSquareErr]
        if len(goodk)>0:
            self._khat = np.min(goodk)
        else:
            self._khat = np.max(self.kRange)
        
        if clear:
            self.clear()
    
    def clear(self):    
        self.x = None
        self.true_labels = None
    

class vn_metrics(object):
    x = None
    n = 0
    
    observed = None
    not_observed = None
    all_not_observed = None
    
    
    dist_to_observed = None
    
    nsrr = 0
    firstCorrect = 0
    
    def __init__(self, x, observed, not_observed):
        self.x = x
        self.observed = observed
        self.not_observed = not_observed
        self.n = x.shape[0]
        
        self.all_not_observed = [v for v in xrange(self.n) if v not in self.observed]
    
    def run(self, clear=True):
        self.distance_to_observed()
        self.compute_nsrr()
        self.first_correct()
        if clear:
            self.clear()
        
    def clear(self):
        self.x = None
        self.observed = None
        self.not_observed = None
        self.all_not_observed = None
        self.dist_to_observed = None

    def nominate_nearest_neighbor(self, k=1):
        dist = np.sum(self.distance_to_observed(),axis=0)
        return np.array([self.all_not_observed[v] for v in np.argsort(dist)[:k]])
    
    def distance_to_observed(self):
        if self.dist_to_observed is None:
            self.dist_to_observed = distance.cdist(
                            self.x[self.observed,:], self.x[self.all_not_observed,:])
        return self.dist_to_observed
    
    def compute_nsrr(self):
        rank = np.zeros(self.n)
        rank[self.all_not_observed] = np.argsort(np.sum(self.distance_to_observed(),axis=0))+1
        n = len(self.not_observed)
        
        self.nsrr = np.sum(1.0/rank[self.not_observed])/np.sum(1.0/np.arange(1,n+1))
        return self.nsrr
    
    def first_correct(self):
        self.firstCorrect = int(self.nominate_nearest_neighbor(1)[0] in self.not_observed)
        return self.firstCorrect
    