import random

import numpy as np
from tqdm import tqdm

from data_parser import get_sets_to_ratings
from plotting import plot_mds, plot_scatter_comparisons
from representativeness import (calculate_representativeness_bayesian,
                                calculate_representativeness_likelihood,
                                calculate_representativeness_max_sim,
                                calculate_representativeness_sum_sim,
                                calculate_songs_to_mds_coords)
from spotify_similarity import calculate_similarity_pairs, create_song_vectors
from matplotlib import pyplot as plt


def evaluate_and_plot(spotify_track_ids, genre_label, audio_vector_keys, random_state = None, plot=False, verbose=True):
    # possible track information: ['album', 'artists', 'available_markets', 'disc_number', 'duration_ms', 'explicit', 'external_ids', 'external_urls', 'href', 'id', 'is_local', 'name', 'popularity', 'preview_url', 'track_number', 'type', 'uri']
    # possible album information: ['album_type', 'artists', 'available_markets', 'external_urls', 'href', 'id', 'images', 'name', 'release_date', 'release_date_precision', 'total_tracks', 'type', 'uri']
    track_vector_keys = ['popularity']
    album_vector_keys = ['release_date']

    sets_to_ratings = get_sets_to_ratings(genre_label=="Pop")

    # print(len([i for i in range(len(sets_to_ratings))]))
    # plt.plot([i for i in range(len(sets_to_ratings))], sets_to_ratings.values(),linestyle='None', marker='.')
    # plt.figure()

    songs_to_song_vectors = create_song_vectors(audio_vector_keys, track_vector_keys, album_vector_keys, spotify_track_ids)
    all_songs = songs_to_song_vectors.keys()
    song_pairs_to_similarity = calculate_similarity_pairs(songs_to_song_vectors)
    songs_to_mds_coords = calculate_songs_to_mds_coords(songs_to_song_vectors, random_state)
    extension_sets = [(("'Dancing Queen' - ABBA", "'I Want It That Way' - Backstreet Boys", "'Congratulations' - Post Malone"), "1 STD Less Representative"),
                      (("'bad guy' - Billie Eilish", "'Just the Way You Are' - Bruno Mars", "'I Want It That Way' - Backstreet Boys"), "1 STD Somewhat Representative"),
                      (("'Single Ladies (Put a Ring on It)' - Beyoncé", "'Congratulations' - Post Malone", "'Just the Way You Are' - Bruno Mars"), "1 STD More Representative")] if genre_label == "Pop" else \
        [(("'WAP (feat. Megan Thee Stallion)' - Cardi B", "'oops!' - Yung Gravy", "'Lucid Dreams' - Juice WRLD"), "1 STD Less Representative"),
         (("'Lose Yourself' - Eminem", "'Drop It Like It's Hot' - Snoop Dogg", "'Lucid Dreams' - Juice WRLD"), "1 STD Somewhat Representative"),
         (("'N.Y. State of Mind' - Nas", "'WAP (feat. Megan Thee Stallion)' - Cardi B", "'Lose Yourself' - Eminem"), "1 STD More Representative")]
    if plot:
        plot_mds(songs_to_mds_coords, genre_label, [(all_songs, "1 STD All Songs")] + extension_sets)

    bayesian_data = []
    max_sim_data = []
    sum_sim_data = []
    likelihood_data = []
    human_data = []

    human_representativeness = []
    for sample_songs in sets_to_ratings:
        bayesian_data.append(calculate_representativeness_bayesian(songs_to_mds_coords, sample_songs))
        max_sim_data.append(calculate_representativeness_max_sim(song_pairs_to_similarity, sample_songs, all_songs))
        sum_sim_data.append(calculate_representativeness_sum_sim(song_pairs_to_similarity, sample_songs, all_songs))
        likelihood_data.append(calculate_representativeness_likelihood(songs_to_mds_coords, sample_songs))
        ratings = sets_to_ratings[sample_songs]
        human_value = sum(ratings)/len(ratings)
        human_data.append(human_value)
        human_representativeness.append((sample_songs, human_value))

    if verbose:
        print(sorted(human_representativeness, key=lambda x:x[1]))
    # best_r_squared, best_label list
    return plot_scatter_comparisons(human_data, bayesian_data, likelihood_data, max_sim_data, sum_sim_data, genre_label, plot=plot, verbose=False)

def test_seeds(pop_track_ids, rap_track_ids, pop_audio_vector_keys, rap_audio_vector_keys):
    list_of_pop_tuples = []
    list_of_rap_tuples = []

    # for k in range(30):
    #     cur_keys = audio_vector_keys.copy()
    #     # pick random subset of features to use 
    #     features_to_keep = np.random.randint(1,len(cur_keys))
    #     random.shuffle(cur_keys)
    #     if features_to_keep + 1 < len(cur_keys):
    #         del cur_keys[features_to_keep + 1:]
    #     cur_keys.sort()
    #     print(k, cur_keys)
    for i in tqdm(range(1000)):
        seed = np.random.randint(0, 100000)
        pop_seed = seed
        rap_seed = seed
        list_of_pop_tuples.append([evaluate_and_plot(pop_track_ids, "Pop", pop_audio_vector_keys, pop_seed, plot=False, verbose=False), pop_seed, pop_audio_vector_keys])
        # plt.figure()
        # list_of_rap_tuples.append([evaluate_and_plot(rap_track_ids, "Rap", rap_audio_vector_keys, rap_seed, plot=False, verbose=False), rap_seed, rap_audio_vector_keys])
        if (i+1)%100 == 0:
            print()
            for list_of_music_tuples in [list_of_pop_tuples]:
                for model in list_of_music_tuples[0][0]:
                    r_values, best_seed, keys_used = max(list_of_music_tuples, key=lambda x:abs(x[0][model]))
                    print("Best r for {}: {:.4f}, {:.2f}, seed: {}, keys: {}".format(model, r_values[model], r_values[model]**2, best_seed, keys_used))
                    # print("    ", r_values)
                r_values, best_seed, keys_used = max(list_of_music_tuples, key=lambda x:abs(x[0]["Bayesian Model"]) + abs(x[0]["Likelihood Model"]))
                print("Best r for Bayesian + Likelihood: {:.4f}, {:.4f}, {:.2f}, {:.2f}, seed: {}, keys: {}".format(r_values["Bayesian Model"], r_values["Likelihood Model"], r_values["Bayesian Model"]**2, r_values["Likelihood Model"]**2, best_seed, keys_used))
                # print("    ", r_values)
                print('---------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------------------------------------------')
    # print(pop_r_values)
    # print(rap_r_values)

def main():
    pop_track_ids = ['0GjEhVFGZW8afUYGChu3Rr', '47BBI51FKFwOMlIiX6m8ya', '7BqBn9nzAq8spo5e7cZ0dJ', '5R9a4t5t5O0IsznsrKPVro', '2Fxmhks0bxGSBdJ92vM42m', '3HWzoMvoF3TQfYg4UPszDq', '3a1lNhkSLSkpJE4MSHpDu9']
    rap_track_ids = ['1SAkL1mYNJlaqnBQxVZrRl', '0trHOzAhNpGCsGBEu7dOJo', '77Ft1RJngppZlq59B6uP0z', '2NBQmPrOEEjA8VbeWOQGxO', '285pBltuF7vW8TeWk8hdRR', '4Oun2ylbjFKMPTiaSbbCih', '7L6G0wpIUiPXuvoo7qhb06']

    # possible audio features: ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature']
    pop_audio_vector_keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    rap_audio_vector_keys = ['acousticness', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'valence']

#      'key', ''mode', ', 'acousticness', 'instrumentalness', 'liveness', 'valence', '',loudness
#   ['acousticness', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'valence']


    # test_seeds(pop_track_ids, rap_track_ids, pop_audio_vector_keys, rap_audio_vector_keys)
    
    evaluate_and_plot(pop_track_ids, "Pop", pop_audio_vector_keys, 38759, plot=True, verbose=False)
    plt.figure()
    evaluate_and_plot(pop_track_ids, "Pop", pop_audio_vector_keys, 44863, plot=True, verbose=False)
    plt.figure()
    # both best at this seed
    evaluate_and_plot(rap_track_ids, "Rap", rap_audio_vector_keys, 58962, plot=True, verbose=False)
    plt.show()

if __name__ == '__main__':
    main()
