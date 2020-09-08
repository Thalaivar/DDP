import numpy as np
import hdbscan
from tqdm import tqdm
from LOCAL_CONFIG import OUTPUT_PATH, MODEL_NAME, HDBSCAN_PARAMS
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

class largeCURE:
    def __init__(self, N, desired_clusters, n_parts, clusters_per_part, soft_cluster=False):
        self.n_parts = n_parts
        self.clusters_per_part = clusters_per_part
        self.soft_clustering = soft_cluster
        self.desired_clusters = desired_clusters

    def process(self, data):
        partitions = self.create_partitions(data)

        logging.info('clustering {} partitions  with {} clusters in each'.format(len(partitions), self.clusters_per_part))
        partition_clusters = []
        pbar = tqdm(total=len(partitions))
        for part in partitions:
            clusterer = cure(part, self.clusters_per_part, number_represent_points=10)
            clusterer.process()
            partition_clusters.append(clusterer)
            pbar.update(1)

        # collect all representative points together
        partial_cluster_repr = []
        partial_cluster_idx = []
        for w in partition_clusters:
            partial_cluster_repr
        with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_CURE.sav'))), 'wb') as f:
            joblib.dump(partition_clusters, f)
        
        
        

        return partition_clusters

    def cluster_partition(self, data):
        n_clusters = np.infty
        cluster_prop = 0.01
        while True:
            min_cluster_size = int(round(cluster_prop * 0.01 * data.shape[0]))
            # cluster partition data
            clusterer = hdbscan.HDBSCAN(min_cluster_size, **HDBSCAN_PARAMS).fit(data)
            # get assignment labels (from soft assignments)
            if self.soft_clustering:
                assignments = np.argmax(hdbscan.all_points_membership_vectors(clusterer), axis=1)
            else:
                assignments = clusterer.labels_
            
            n_clusters = assignments.max()
            # break if number of clusters is less than desired clusters/partition but greater than 3 * required clusters
            if n_clusters <= self.clusters_per_part and n_clusters >= 3 * self.desired_clusters:
                break
            else:
                logging.info('partition_clusters: {}, clusters/partition: [{}, {}]'.format(n_clusters, 3 * self.desired_clusters, self.clusters_per_part))
                # if clusters are too less, decrease min_cluster_size
                if n_clusters < 2 * self.desired_clusters:
                    if cluster_prop < HDBSCAN_PARAMS['min_samples']:
                        logging.warning('maximum possible clusters less than 2 * desired_clusters')
                        break
                    cluster_prop *= 0.8
                    logging.debug('decreasing `min_cluster_size`')
                # if clusters are too many, increase min_cluster_size
                elif n_clusters > self.clusters_per_part:
                    cluster_prop *= 1.2
                    logging.debug('increasing `min_cluster_size`')

        logging.info('identified {} clusters'.format(n_clusters))
        # create cluster objects
        clusters = []
        logging.info('creating clusters with respective data points')
        for label in np.unique(assignments):
            idx = np.where(assignments == label)[0]
            clusters.append(Cluster(data[idx,:]))
        
        # calculate representative points for each cluster
        logging.info('calculating representative points for each cluster')
        for cluster in clusters:
            cluster.create_representative_points()
            
        return clusters

    def create_partitions(self, data):
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

class Cluster:
    def __init__(self, data: np.ndarray, n_rep: int=10, alpha: float=0.5):
        self.alpha = alpha
        self.n_rep = n_rep
        self.data = data

        # data for CURE
        self.mean = self.get_mean()
        self.rep = None

    def create_representative_points(self):
        scattered_points = well_scattered_points(self.n_rep, self.mean, self.data)

        # shrink points toward mean
        rep = np.array([p + self.alpha*(self.mean - p) for p in scattered_points])
        self.rep = rep
        return rep

    def get_mean(self):
        return np.sum(self.data, axis=0)/self.data.shape[0]
    
    def scattered_points_from_rep(self):
        return (self.rep - self.alpha*self.mean)/(1 - self.alpha)

    @property
    def size(self):
        return self.data.shape[0]

def cluster_fast_merge(u: Cluster, v: Cluster):
    combied_data = np.vstack((u.data, v.data))
    # assuming that both u, v have same alpha and n_rep
    w = Cluster(combied_data, u.n_rep, u.alpha)

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
    scattered_points = well_scattered_points(w.n_rep, w.mean, scattered_points)

    # get representative points for merged cluster by shrinking
    w.rep = np.array([p + w.alpha*(w.mean - p) for p in scattered_points])

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

# def dist(u: Cluster, v: Cluster):
