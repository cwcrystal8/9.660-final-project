import math

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.manifold import MDS


def calculate_representativeness_bayesian(songs_to_mds_coords, sample_songs):
    # Nlog S −N(m−μ)TV−1(m−μ) −trace(SV−1 )
    # 2 x num sample songs
    sample_songs_np_coords = np.asarray([np.asarray(songs_to_mds_coords[s]) for s in sample_songs]).T
    all_songs_np_coords = np.asarray([np.asarray(songs_to_mds_coords[s]) for s in songs_to_mds_coords]).T
    # 1x1
    N = len(sample_songs)
    # 2x1
    m = np.sum(sample_songs_np_coords, axis=1, keepdims=True)/N
    # 2x2
    # S = np.sum([(coords - m).T*(coords-m) for coords in sample_songs_np_coords], axis=1)
    # 2 x num sample songs
    temp = sample_songs_np_coords-m
    S = sum([np.dot(temp[:, i:i+1], temp[:, i:i+1].T) for i in range(sample_songs_np_coords.shape[1])])
    # 2x1
    mu = np.sum(all_songs_np_coords, axis=1, keepdims=True)/len(songs_to_mds_coords)
    # 2x2 
    V_inv = np.linalg.inv(np.cov(all_songs_np_coords)) # covariance of all songs? 
    output =  (N*np.log(np.linalg.det(S)) - N*np.dot(np.dot((m-mu).T, V_inv), (m-mu)) - np.trace(np.dot(S,V_inv)))[0][0]
    return output

def calculate_representativeness_likelihood(songs_to_mds_coords, sample_songs):
    # num sample songs x 2
    all_songs_np_coords = np.asarray([np.asarray(songs_to_mds_coords[s]) for s in songs_to_mds_coords])
    # 1x2
    mu = np.sum(all_songs_np_coords, axis=0)/len(songs_to_mds_coords)
    # 2x2
    cov = np.cov(all_songs_np_coords.T)
    dist = multivariate_normal(mu, cov)
    # each 1x2
    likelihoods = [dist.pdf(np.asarray(songs_to_mds_coords[s])) for s in sample_songs]
    return math.prod(likelihoods)

def calculate_representativeness_max_sim(song_pairs_to_similarity, sample_songs, all_songs):
    total = 0
    for song_j in all_songs:
        similarities = []
        for song_i in sample_songs:
            similarities.append(song_pairs_to_similarity[frozenset((song_j, song_i))])
        total += max(similarities)
    return total

def calculate_representativeness_sum_sim(song_pairs_to_similarity, sample_songs, all_songs):   
    total = 0
    for song_j in all_songs:
        for song_i in sample_songs:
            total += song_pairs_to_similarity[frozenset((song_j, song_i))]
    return total

def calculate_songs_to_mds_coords(songs_to_song_vectors):
    # we choose a 2D feature space as in the paper
    embedding = MDS(n_components=2)
    features = []
    labels = []
    for song in songs_to_song_vectors:
        features.append(songs_to_song_vectors[song])
        labels.append(song)
    mds_coords = embedding.fit_transform(features)
    return {labels[i]: mds_coords[i] for i in range(len(labels))}
