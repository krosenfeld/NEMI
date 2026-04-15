'''
Test basic functionality of the method
'''

import pytest
import numpy as np
from nemi import NEMI

def test_micro_nemi_pack():
    '''
    Run the method on noise for an ensemble
    '''

    X = np.random.random((100,5))

    nemi  = NEMI()
    nemi.run(X, n=3)
    nemi.plot('clusters')

    

def test_micro_nemi():
    '''
    Run the method on noise for single member
    '''

    X = np.random.random((100,5))

    nemi  = NEMI()
    nemi.run(X, n=1)
    nemi.plot('clusters')


def test_entropy_selected_base():
    '''
    Run ensemble mode with entropy-based base selection.
    '''

    rng = np.random.default_rng(42)
    X = rng.random((100, 5))

    params = dict(
        embedding_dict=dict(min_dist=0.0, n_components=3, n_neighbors=10),
        clustering_dict=dict(linkage='ward', n_clusters=4, n_neighbors=10),
        ensemble_dict=dict(base_selection='min_entropy', base_id=0, max_clusters=None),
    )

    nemi = NEMI(params=params)
    nemi.run(X, n=3)

    assert nemi.clusters.shape == (X.shape[0],)
    assert 0 <= nemi.base_id < len(nemi.nemi_pack)
    assert nemi.entropy.shape == (X.shape[0],)
    assert nemi.entropy_by_base.shape == (len(nemi.nemi_pack), X.shape[0])
    assert nemi.base_scores.shape == (len(nemi.nemi_pack),)
    assert np.all(np.isfinite(nemi.entropy))
    assert np.all((nemi.entropy >= 0) & (nemi.entropy <= 100))
