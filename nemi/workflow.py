
import umap
import pickle
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
# import sciris as sc

__all__ = ['NEMI', 'SingleNemi']

default_params = dict(
    embedding_dict = dict(min_dist=0.0, n_components=3, n_neighbors=20),
    clustering_dict = dict(linkage='ward',  n_clusters=30, n_neighbors=40),
    ensemble_dict = dict(base_selection='fixed', base_id=0, max_clusters=None)
)


def _merge_params(params=None):
    """Merge user parameters into the nested default parameter dictionary."""

    merged = copy.deepcopy(default_params)
    for key, value in (params or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value

    return merged


class SingleNemi():
    """
    A single instance of the NEMI pipeline.

    Args:
        params (dict, optional): Nested parameter dictionary for the workflow.
            Supported keys are ``embedding_dict``, ``clustering_dict``, and
            ``ensemble_dict``. Missing values fall back to
            ``nemi.workflow.default_params``.
    """

    def __init__(self, params=None):

        # pipeline parameters
        # self.params = sc.mergedicts(default_params, params)
        # pipeline parameters
        self.params = _merge_params(params)

        # set during the run
        self.embedding = None
        self.clusters = None
        self.X = None

        return
    
    def run(self, X, save_steps=True):
        """ Run a single instance of the NEMI pipeline

        The pipeline consists of steps: 
        
        - fitting the embedding
        - predicting the clusters, 
        - sorting the clusters by descending size

        Args:
            X (:py:class:`~numpy.ndarray`): The data contained in a sparse matrix of shape (``n_samples``, ``n_features``)
        """

        # fit the embedding
        print('Fitting the embedding')
        self.fit_embedding(X)

        # predict the clusters
        print('Predicting the clusters')
        self.clusters = self.predict_clusters()

        # sort the clusters by (descending) size
        print('Sorting clusters')
        self.clusters = self.sort_clusters(self.clusters)

    def scale_data(self, X):
        """ Scale the data to have a mean and variance of 1.

        Args:
            X (:py:class:`~numpy.ndarray`): The data to pick seeds for. A sparse matrix of shape (``n_samples``, ``n_features``)
            **kwargs : keyword arguments to embedding function
        """

        # scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X)
        return scaled_data

    def fit_embedding(self, X):
        """ Run the embedding algorithm on the data

        Args
            X (:py:class:`~numpy.ndarray`): The data to pick seeds for. A sparse matrix of shape (``n_samples``, ``n_features``)
            **kwargs : keyword arguments to embedding function
        """

        # initialize data
        self.X = X
        # run embedding
        self.embedding = self.__embedding_algo(**self.params['embedding_dict'])(self.X)


    def predict_clusters(self):
        """ Run the clustering algorithm on the embedding

        Clustering algorithm parameters is set by the ``clustering_dict`` attribute.

        Returns:
            Identified clusters
        """

        return self.__clustering_algo(**self.params['clustering_dict'])(self.X)


    def sort_clusters(self, clusters):
        """ Updates cluster labels 0,1,...,k so that each cluster is of descending size.

        Args:
            clusters (:py:class`~numpy.ndarray`, list)

        Returns:
            An array with the new labels
        """

        # number of clusters (also the same as the label name in the agglomerated cluster dict)
        n_clusters = np.max(clusters)+1
        #  create a histogram of the different clusters
        hist,_ = np.histogram(clusters, np.arange(n_clusters+1))
        # clusters sorted by size (largest to smallest)
        sorted_clusters= np.argsort(hist)[::-1]
        # assign new labels where labels 0,...,k go in decreasing member size 
        new_labels = np.empty(clusters.shape)
        new_labels.fill(np.nan)
        for new_label, old_label in enumerate(sorted_clusters):
            new_labels[clusters == old_label] = new_label

        return new_labels
        
    def save(self, filename):
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)

    def load_embedding(self, filename):
        self.embedding = np.load(filename)

    def save_embedding(self, filename):
        """ Save the embedding to a file

        Args:
            filename (str): Filename to save embedding
        """
        np.save(filename, self.embedding)

    def plot(self, to_plot=None, **kwargs):
        if to_plot.lower() == 'embedding':
            self._plot_embedding(**kwargs)
        elif to_plot.lower() == 'clusters':
            self._plot_clusters(**kwargs)

    def _plot_embedding(self, s=1, subsample=10, alpha=0.4):

        data = self.embedding

        fig = plt.figure()
        if data.shape[1] == 2:
            ax = plt.gca()
        elif data.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
        else:
            raise RuntimeError('Embedding not consistent with plotting function')

        ax.scatter(*data[::subsample].T, s=s, alpha=alpha, zorder=4)

    def _plot_clusters(self, n=None, s=1, subsample=10, alpha=0.4):

        self._plot_embedding(s=s, subsample=subsample, alpha=alpha)

        data = self.embedding
        ax = plt.gca()
        labels = self.clusters
        unique_labels = np.sort(np.unique(labels))
        colors = [plt.cm.tab20(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            class_member_mask = (labels == k)
            xy = data[class_member_mask, :]
            ax.scatter(*xy[::subsample].T, c=np.array(col).reshape((1,-1)), s=s, alpha=1, zorder=4)      


    def __embedding_algo(self, **kwargs):
        return umap.UMAP(**kwargs).fit_transform

    def __clustering_algo(self, **kwargs):
        """ Clustering step

        Args:
            n_neighbors (int): Number of neighbors for each sample of the kneighbors_graph. Defaults to 40.
                   
        """
        # Create a graph capturing local connectivity. Larger number of neighbors
        # will give more homogeneous clusters to the cost of computation
        # time. A very large number of neighbors gives more evenly distributed
        # cluster sizes, but may not impose the local manifold structure of
        # the data
        knn_graph = kneighbors_graph(self.embedding, kwargs['n_neighbors'], include_self=False)
        model = AgglomerativeClustering(linkage=kwargs['linkage'],
                                            connectivity=knn_graph,
                                            n_clusters=kwargs['n_clusters'])
        return model.fit_predict                          


class NEMI(SingleNemi):
    """ Main NEMI workflow.

    Args:
        params (dict, optional): Nested parameter dictionary for the workflow.
            Use ``embedding_dict`` and ``clustering_dict`` to configure a single
            run, and ``ensemble_dict`` to control how ensemble members are
            compared when ``n > 1``.
    """

    def __init__(self, params=None):
        # pipeline parameters
        self.params = _merge_params(params)
        self.embedding = None
        self.clusters = None
        self.X = None
        self.nemi_pack = []
        self.base_id = None
        self.base_selection = None
        self.base_scores = None
        self.sorted_overlap = None
        self.agg_overlaps = None
        self.entropy = None
        self.entropy_by_base = None

    def run(self, X, n=1):
        """ Run the NEMI pipeline.

        A single run fits the embedding, predicts clusters, and sorts cluster
        labels by descending size. When ``n > 1``, NEMI runs an ensemble of
        stochastic realisations and combines them using the strategy from
        ``params['ensemble_dict']``.

        Args:
            X (:py:class:`~numpy.ndarray`): The data contained in a sparse matrix of shape (``n_samples``, ``n_features``).
            n (int, optional): Number of iterations to run. Defaults to 1.

        Notes:
            Ensemble runs populate ``base_id``, ``base_selection``,
            ``sorted_overlap``, ``agg_overlaps``, ``entropy``,
            ``entropy_by_base``, and ``base_scores`` on the workflow instance.
        """
        self.X = X

        if n == 1:
            super().run(X)
            return
        else:
            # initialize the pack
            nemi_pack = []
            # run the pack
            for member in tqdm(np.arange(n)):
                # create nemi instance
                nemi = SingleNemi(params=self.params)
                # run single instance
                nemi.run(X)        
                # add to the pack
                nemi_pack.append(nemi)

            self.nemi_pack = nemi_pack

        self.assess_overlap(**self.params['ensemble_dict'])

    def plot(self, to_plot=None, plot_ensemble=False, **kwargs):

        if plot_ensemble:
            for nemi in self.nemi_pack:
                nemi.plot(to_plot, **kwargs)

        if to_plot == 'clusters':
            super().plot('clusters')

    def _relabel_overlaps(self, base_id:int =0, max_clusters=None):
        """Relabel all ensemble members against a selected base member."""

        self._validate_base_id(base_id)

        compare_ids = [i for i in range(len(self.nemi_pack))]
        compare_ids.pop(base_id)

        base_labels = self.nemi_pack[base_id].clusters
        num_clusters = int(np.max(base_labels) + 1)

        if max_clusters is None:
            max_clusters = num_clusters

        if max_clusters < 1 or max_clusters > num_clusters:
            raise ValueError(f'max_clusters must be between 1 and {num_clusters}')

        sorted_overlap = np.zeros((len(compare_ids)+1, max_clusters, base_labels.shape[0])) * np.nan

        data_vector = [nemi.clusters for id, nemi in enumerate(self.nemi_pack) if id != base_id]

        for compare_cnt, compare_id in enumerate(compare_ids):
            compare_labels = data_vector[compare_cnt]
            summary_stats = np.zeros((num_clusters, max_clusters))

            for c1 in range(max_clusters):
                data1_M = np.zeros(base_labels.shape, dtype=int)
                data1_M[np.where(c1 == base_labels)] = 1

                for c2 in range(num_clusters):
                    data2_M = np.zeros(base_labels.shape, dtype=int)
                    data2_M[np.where(c2 == compare_labels)] = 1

                    num_overlap = np.sum(data1_M * data2_M)
                    num_total = np.sum(data1_M | data2_M)
                    summary_stats[c2, c1] = (num_overlap / num_total) * 100

            used_clusters = set()
            for c1 in range(max_clusters):
                sorted_overlap_for_one_cluster = np.zeros(base_labels.shape, dtype=int) * np.nan
                sorted_clusters = np.argsort(summary_stats[:, c1])[::-1]
                biggest_cluster = [ele for ele in sorted_clusters if ele not in used_clusters][0]
                used_clusters.add(biggest_cluster)

                data2_M = np.zeros(base_labels.shape, dtype=int)
                data2_M[np.where(biggest_cluster == compare_labels)] = 1

                sorted_overlap_for_one_cluster[np.where(data2_M == 1)] = 1
                sorted_overlap[compare_id, c1, :] = sorted_overlap_for_one_cluster

        for c1 in range(max_clusters):
            sorted_overlap[base_id, c1, :] = 1 * (base_labels == c1)

        return sorted_overlap

    def _entropy_from_sorted_overlap(self, sorted_overlap):
        """Return normalized per-sample Shannon entropy from relabelled overlaps."""

        relabelled_clusters = np.argmax(np.nan_to_num(sorted_overlap), axis=1).astype(int)
        max_clusters = sorted_overlap.shape[1]
        counts = np.zeros((relabelled_clusters.shape[1], max_clusters), dtype=float)

        for cluster_id in range(max_clusters):
            counts[:, cluster_id] = np.sum(relabelled_clusters == cluster_id, axis=0)

        totals = counts.sum(axis=1, keepdims=True)
        probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        log_probabilities = np.zeros_like(probabilities)
        positive_probabilities = probabilities > 0
        log_probabilities[positive_probabilities] = np.log2(probabilities[positive_probabilities])
        entropy = -(probabilities * log_probabilities).sum(axis=1)

        entropy_max = np.log2(max_clusters)
        if entropy_max == 0:
            return np.zeros(entropy.shape[0])

        return (entropy * 100) / entropy_max

    def _entropy_for_base(self, base_id:int =0, max_clusters=None):
        """Calculate relabelled overlaps and entropy for one base member."""

        sorted_overlap = self._relabel_overlaps(base_id=base_id, max_clusters=max_clusters)
        entropy = self._entropy_from_sorted_overlap(sorted_overlap)
        return entropy, sorted_overlap

    def _majority_vote(self, sorted_overlap):
        """Aggregate relabelled overlaps with a majority vote."""

        agg_overlaps = np.nansum(sorted_overlap, axis=0)
        vote_overlaps = np.argmax(agg_overlaps, axis=0)
        return agg_overlaps, vote_overlaps

    def _validate_base_id(self, base_id):
        if base_id < 0 or base_id >= len(self.nemi_pack):
            raise ValueError(f'base_id must be between 0 and {len(self.nemi_pack) - 1}')

    def assess_overlap(self, base_id:int =0, max_clusters=None, base_selection='fixed', **kwargs):
        """Assess agreement across an ensemble and export consensus cluster labels.

        Args:
            base_id (int, optional): Index of ensemble member to use as the base comparison.
            max_clusters (int, optional): Number of sorted clusters to compare. When
                ``None``, use all sorted clusters from the selected base member.
            base_selection (str, optional): Strategy used to choose the base member.
                Use ``'fixed'`` to keep ``base_id`` or ``'min_entropy'`` to choose
                the member with the lowest mean relabelled entropy.

        Returns:
            None. Consensus labels are written to ``self.clusters``.

        Notes:
            This method also stores ensemble diagnostics on the instance,
            including ``base_id``, ``base_selection``, ``sorted_overlap``,
            ``agg_overlaps``, ``entropy``, ``entropy_by_base``, and
            ``base_scores``.
        """

        if not self.nemi_pack:
            raise RuntimeError('Cannot assess overlap before running an ensemble (n > 1)')

        if base_selection == 'fixed':
            self._validate_base_id(base_id)
            sorted_overlap = self._relabel_overlaps(base_id=base_id, max_clusters=max_clusters)
            entropy = self._entropy_from_sorted_overlap(sorted_overlap)
            entropy_by_base = None
            base_scores = None
        elif base_selection == 'min_entropy':
            entropy_by_base = []
            base_scores = []
            sorted_overlaps = []

            for candidate_base_id in range(len(self.nemi_pack)):
                candidate_entropy, candidate_overlap = self._entropy_for_base(
                    base_id=candidate_base_id,
                    max_clusters=max_clusters,
                )
                entropy_by_base.append(candidate_entropy)
                base_scores.append(candidate_entropy.mean())
                sorted_overlaps.append(candidate_overlap)

            entropy_by_base = np.vstack(entropy_by_base)
            base_scores = np.asarray(base_scores)
            base_id = int(np.argmin(base_scores))
            entropy = entropy_by_base[base_id]
            sorted_overlap = sorted_overlaps[base_id]
        else:
            raise ValueError("base_selection must be 'fixed' or 'min_entropy'")

        agg_overlaps, vote_overlaps = self._majority_vote(sorted_overlap)

        self.base_id = base_id
        self.base_selection = base_selection
        self.embedding = self.nemi_pack[base_id].embedding
        self.sorted_overlap = sorted_overlap
        self.agg_overlaps = agg_overlaps
        self.entropy = entropy
        self.entropy_by_base = entropy_by_base
        self.base_scores = base_scores
        self.clusters = vote_overlaps
