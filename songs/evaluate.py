from data_parser import get_sets_to_ratings
from plotting import plot_mds, plot_scatter_comparisons
from representativeness import (calculate_representativeness_bayesian,
                                calculate_representativeness_likelihood,
                                calculate_representativeness_max_sim,
                                calculate_representativeness_sum_sim,
                                calculate_songs_to_mds_coords)
from spotify_similarity import calculate_similarity_pairs, create_song_vectors
from matplotlib import pyplot as plt


def evaluate_and_plot(spotify_track_ids, genre_label):
    # possible audio features: ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature']
    audio_vector_keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    # possible track information: ['album', 'artists', 'available_markets', 'disc_number', 'duration_ms', 'explicit', 'external_ids', 'external_urls', 'href', 'id', 'is_local', 'name', 'popularity', 'preview_url', 'track_number', 'type', 'uri']
    # possible album information: ['album_type', 'artists', 'available_markets', 'external_urls', 'href', 'id', 'images', 'name', 'release_date', 'release_date_precision', 'total_tracks', 'type', 'uri']
    track_vector_keys = ['popularity']
    album_vector_keys = ['release_date']

    sets_to_ratings = get_sets_to_ratings(genre_label=="Pop")

    songs_to_song_vectors = create_song_vectors(audio_vector_keys, track_vector_keys, album_vector_keys, spotify_track_ids)
    all_songs = songs_to_song_vectors.keys()
    song_pairs_to_similarity = calculate_similarity_pairs(songs_to_song_vectors)
    songs_to_mds_coords = calculate_songs_to_mds_coords(songs_to_song_vectors)
    plot_mds(songs_to_mds_coords, genre_label, True)

    bayesian_data = []
    max_sim_data = []
    sum_sim_data = []
    likelihood_data = []
    human_data = []

    for sample_songs in sets_to_ratings:
        bayesian_data.append(calculate_representativeness_bayesian(songs_to_mds_coords, sample_songs))
        max_sim_data.append(calculate_representativeness_max_sim(song_pairs_to_similarity, sample_songs, all_songs))
        sum_sim_data.append(calculate_representativeness_sum_sim(song_pairs_to_similarity, sample_songs, all_songs))
        likelihood_data.append(calculate_representativeness_likelihood(songs_to_mds_coords, sample_songs))
        ratings = sets_to_ratings[sample_songs]
        human_data.append(sum(ratings)/len(ratings))

    plot_scatter_comparisons(human_data, bayesian_data, likelihood_data, max_sim_data, sum_sim_data, genre_label)

def main():
    pop_track_ids = ['0GjEhVFGZW8afUYGChu3Rr', '47BBI51FKFwOMlIiX6m8ya', '7BqBn9nzAq8spo5e7cZ0dJ', '5R9a4t5t5O0IsznsrKPVro', '2Fxmhks0bxGSBdJ92vM42m', '3HWzoMvoF3TQfYg4UPszDq', '3a1lNhkSLSkpJE4MSHpDu9']
    rap_track_ids = ['1SAkL1mYNJlaqnBQxVZrRl', '0trHOzAhNpGCsGBEu7dOJo', '77Ft1RJngppZlq59B6uP0z', '2NBQmPrOEEjA8VbeWOQGxO', '285pBltuF7vW8TeWk8hdRR', '4Oun2ylbjFKMPTiaSbbCih', '7L6G0wpIUiPXuvoo7qhb06']

    evaluate_and_plot(pop_track_ids, "Pop")
    plt.figure()
    evaluate_and_plot(rap_track_ids, "Rap")
    plt.show()

if __name__ == '__main__':
    main()
