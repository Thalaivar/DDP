import logging
import hdbscan
import itertools
import joblib
import heapq
import numpy as np
import time
from tqdm import tqdm
from kdtree import kdtree

HDBSCAN_PARAMS = {'min_samples': 10, 'prediction_data': True}

class Cluster:
    def __init__(self, data: np.ndarray, n_rep=int(1e5), alpha=0.5, calc_mean=True):
        self.alpha = alpha
        self.n_rep = n_rep
        self.data = data

        self.mean = self.get_mean() if calc_mean else None
        self.rep = None

        # for use with CURE
        self.closest = None
        self.distance = np.inf
        self.rep_idx = []
    
    def create_representative_points(self):
        scattered_points = well_scattered_points(self.n_rep, self.mean, self.data)
        # shrink points toward mean
        rep = np.array([p + self.alpha*(self.mean - p) for p in scattered_points])        
        self.rep = rep
        return rep
    
    def get_mean(self):
        return self.data.mean(axis=0)
    
    def scattered_points_from_rep(self):
        return (self.rep - self.alpha*self.mean)/(1 - self.alpha)
    
    def __lt__(self, c):
        return self.distance < c.distance

    @property
    def __len__(self):
        return self.data.shape[0]

    @property
    def dims(self):
        return self.data.shape[1]

    @staticmethod
    def _distance(u, v):
        min_dist = np.inf
        for i in range(u.rep.shape[0]):
            for j in range(v.rep.shape[0]):
                dist = np.linalg.norm(u.rep[i] - v.rep[j])
                min_dist = dist if dist < min_dist else min_dist
        
        return min_dist

    @staticmethod
    def _cluster_fast_merge(u, v):
        combined_data = np.vstack((u.data, v.data))
        # assuming that both u, v have same alpha and n_rep
        w = Cluster(combined_data, u.n_rep, u.alpha, calc_mean=False)

        # check that clusters have their means calculate
        if u.mean is None or v.mean is None:
            raise ValueError('means of clusters need to be calculated for merging')
        # get new mean
        w.mean = (u.size*u.mean + v.size*v.mean)/(u.size + v.size)

        # check that both clusters have representative points assigned
        if u.rep is None or v.rep is None:
            raise ValueError('representative points of clusters need to be assigned before merging')
        # get "well scattered" points from both clusters
        scattered_points = np.vstack((u.scattered_points_from_rep(),
                                    v.scattered_points_from_rep()))
        scattered_points = well_scattered_points(w.n_rep, w.mean, w.data)
        # get representative points for merged cluster by shrinking
        w.rep = np.array([p + w.alpha*(w.mean - p) for p in scattered_points])
        
        # assign representative point IDs to new cluster
        w.rep_idx.extend(u.rep_idx)
        w.rep_idx.extend(v.rep_idx)

        return w

def well_scattered_points(n_rep: int, mean: np.ndarray, data: np.ndarray):
    # if the cluster contains less than no. of rep points, all points are rep points
    if data.shape[0] <= n_rep:
        return data

    # get well scattered points
    tmp_set = []
    for i in range(n_rep):
        max_dist = 0
        for j in range(data.shape[0]):
            p = data[j]
            if i == 0:
                min_dist = np.linalg.norm(p - mean)
            else:
                min_dist = np.min(np.linalg.norm(p - np.array(tmp_set), axis=1))
            
            if min_dist >= max_dist:
                max_dist = min_dist
                max_point = p
        tmp_set.append(max_point)
    
    return tmp_set

class CURE:
    def __init__(self, desired_clusters):
        self.desired_clusters = desired_clusters
        self.init_w_clusters = False

        self.heap_ = None
        self.tree_ = None
        self.rep_cluster_idx = None

    def process(self, clusters: list):
        self._create_heap(clusters)
        self._create_kdtree()

        iteration = 0
        while len(self.heap_) > self.desired_clusters:
            logging.info('iteration {} with {} clusters'.format(iteration, len(self.heap_)))

            u = heapq.heappop(self.heap_)
            v = u.closest

            self.heap_.remove(u)
            self.heap_.remove(v)

            self._delete_rep_points(u)
            self._delete_rep_points(v)

            w = Cluster._cluster_fast_merge(u, v)
            
            self._insert_rep_points(w)

            cluster_dist_changed = 0
            recalc_time_start = time.time()
            for c in self.heap_:
                dist = Cluster._distance(w, c)
                if  dist < w.distance:
                    w.closest = c
                
                if c.closest is u or c.closest is v:
                    if c.distance < dist:
                        (c.closest, c.distance) = self._find_closest_cluster(c, dist)
                        cluster_dist_changed += 1
                    
                    if c.closest is None:
                        c.closest = w
                        c.distance = dist

            recalc_time = time.time() - recalc_time_start
            logging.debug('{} cluster distances recalculated in {:.4f} s'.format(cluster_dist_changed, recalc_time))
            heapq.heapify(self.heap_)
            heapq.heappush(self.heap_, w)
            
            iteration += 1

        return self.heap_

    def _create_heap(self, clusters: list):
        self.heap_ = clusters
        for cluster in self.heap_:
            for i in range(len(self.heap_)):
                if self.heap_[i] != cluster:
                    dist = Cluster._distance(self.heap_[i], cluster)
                    if dist  < cluster.distance:
                        cluster.closest = self.heap_[i]
                        cluster.distance = dist

        # create heap of clusters sorted by closest distance
        heapq.heapify(self.heap_)
    
    def _create_kdtree(self):
        representatives, payloads = [], []
        for current_cluster in self.heap_:
            for representative_point in current_cluster.rep:
                representatives.append(representative_point)
                payloads.append(current_cluster)

        # initialize it using constructor to have balanced tree at the beginning to ensure the highest performance
        # when we have the biggest amount of nodes in the tree.
        self.tree_ = kdtree(representatives, payloads)

    def _find_closest_cluster(self, cluster: Cluster, distance: float):
        nearest_cluster = None
        min_dist = np.inf

        real_euclidean_distance = distance ** 0.5

        for point in cluster.rep:
            # Nearest nodes should be returned (at least it will return itself).
            nearest_nodes = self.tree_.find_nearest_dist_nodes(point, real_euclidean_distance)
            for (candidate_distance, kdtree_node) in nearest_nodes:
                if (candidate_distance < min_dist) and (kdtree_node is not None) and (kdtree_node.payload is not cluster):
                    min_dist = candidate_distance
                    nearest_cluster = kdtree_node.payload
        
        return (nearest_cluster, min_dist)

        
    def _delete_rep_points(self, cluster):
        for point in cluster.rep:
            self.tree_.remove(point, payload=cluster)
    
    def _insert_rep_points(self, cluster):
        for point in cluster.rep:
            self.tree_.insert(point, cluster)

class bigCURE(CURE):
    def __init__(self, desired_clusters, n_parts, clusters_per_part, soft_cluster=False):
        super().__init__(desired_clusters)
        self.n_parts = n_parts
        self.clusters_per_part = clusters_per_part
        self.soft_clustering = soft_cluster

        self.prev_min_cluster_prop = None
    
    def process(self, data: np.ndarray):
        partitions = self._partition_dataset(data)
        
        # pre-cluster each partition
        logging.info('pre-clustering {} partitions to have {} clusters each'.format(len(partitions), self.clusters_per_part))
        partition_clusters = []
        for i in tqdm(range(len(partitions))):
            partition_clusters.extend(self._cluster_partition(partitions[i]))
        
        logging.info('total {} clusters identified in pre-clustering'.format(len(partition_clusters)))

        # delete after final
        with open('checkpoint1.sav', 'wb') as f:
            joblib.dump(partition_clusters, f)

        logging.info('second clustering pass using CURE on {} clusters'.format(len(partition_clusters)))
        clusters = super().process(partition_clusters)

        return clusters


    def _cluster_partition(self, data: np.ndarray, create_clusters=True):
        min_clusters = 3 * self.desired_clusters
        min_cluster_prop = 0.01 if self.prev_min_cluster_prop is None else self.prev_min_cluster_prop
        while True:
            min_cluster_size = int(round(min_cluster_prop * 0.01 * data.shape[0]))
            # cluster partition data
            clusterer = hdbscan.HDBSCAN(min_cluster_size, **HDBSCAN_PARAMS).fit(data)
            
            # get assignment labels (from soft assignments)
            if self.soft_clustering:
                assignments = np.argmax(hdbscan.all_points_membership_vectors(clusterer), axis=1)
            else:
                assignments = clusterer.labels_
            
            n_clusters = assignments.max() + 1
            # break if number of clusters is within desired range
            if n_clusters <= self.clusters_per_part and n_clusters > min_clusters:
                self.prev_min_cluster_prop = min_cluster_prop
                break
            else:
                logging.debug('partition clusters: {}; required partition clusters: [{},{}]'.format(n_clusters, min_clusters, self.clusters_per_part))

                if n_clusters < min_clusters:
                    if min_cluster_size < HDBSCAN_PARAMS['min_samples']:
                        # minimum cluster size cannot be reduced any futher -> maximum number of samples identified
                        logging.warning('maximum possible clusters in partition: {}, less than minimum clusters required: {}'.format(n_clusters, min_clusters))
                        break
                    # reduce minimum cluster size to increase number of clusters
                    min_cluster_prop *= 0.8
                    logging.debug('decreasing `min_cluster_size`')
                
                elif n_clusters > self.clusters_per_part:
                    # increase minimum cluster size to decrease number of clusters
                    min_cluster_prop *= 1.2
                    logging.debug('increasing `min_cluster_size`')
        
        logging.debug('identified {} clusters for partition with samples {}'.format(n_clusters, data.shape[0]))
        
        if create_clusters:
            # create cluster objects from assignments
            labels = np.unique(assignments)
            clusters = [[] for i in labels if i >= 0]
            for i, label in enumerate(assignments):
                if label >= 0:
                    clusters[label].append(data[i])
            clusters = [Cluster(np.array(cdata)) for cdata in clusters]

            logging.debug('created clusters from assignments and calculating representative points')
            for cluster in clusters:
                cluster.create_representative_points()

            return clusters, assignments
        else:
            return assignments
            
    def _partition_dataset(self, data: np.ndarray):
        partitions = []
        batch_sz = data.shape[0] // self.n_parts
        
        logging.info('creating {} partitions of {} samples for batch size {}'.format(self.n_parts, data.shape[0], batch_sz))

        i = 0
        for _ in range(self.n_parts):
            if i + batch_sz < data.shape[0] - 1:
                partitions.append(data[i:i+batch_sz,:])
                i += batch_sz
            else:
                partitions.append(data[i:,:])
        
        return partitions