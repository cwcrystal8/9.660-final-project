import itertools

import spotipy
from scipy import spatial
from spotipy.oauth2 import SpotifyClientCredentials

CID = 'MYCID'
SECRET = 'MYSECRET'

def create_song_vectors(audio_vector_keys, track_vector_keys, album_vector_keys, song_ids):
    client_credentials_manager = SpotifyClientCredentials(client_id=CID, client_secret=SECRET)
    spotify = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
    songs_to_song_vectors = {}
    for song_id in song_ids:
        audio_features = spotify.audio_features(song_id)[0]
        track = spotify.track(song_id)
        vector = []
        for k in audio_vector_keys:
            value = audio_features[k]
            if k == 'key':
                if value == -1:
                    raise Exception("Could not find key")
                # bottom of the range is 0 
                # range is 11
                value /= 11
            if k == 'loudness':
                # bottom of the range is -60
                # range is 60
                value += 60
                value /= 60
            if k == 'tempo':
                # bottom of range of our songs is around 83, we'll do 50, top is 193, we'll do 200 
                value -= 50
                value /= 150
            if k == 'duration_ms':
                # 139760 to 320627
                value -= 130000
                value /= 200000
            vector.append(value)
        for k in track_vector_keys:
            value = track[k]
            if k == 'popularity':
                # range is 0-100
                value /= 100
            vector.append(value)
        for k in album_vector_keys:
            value = track['album'][k]
            if k == 'release_date':
                value = int(value[:4])
                # range from 1976-2020
                value -= 1970
                value /= 50
            vector.append(value)
        songs_to_song_vectors['\'{}\' - {}'.format(track['name'], track['artists'][0]['name'])] = vector
    return songs_to_song_vectors

def calculate_similarity_pairs(songs_to_song_vectors):
    similarities = {}
    for song1, song2, in itertools.combinations(songs_to_song_vectors.keys(), 2):
        similarities[frozenset((song1, song2))] = 1 - spatial.distance.cosine(songs_to_song_vectors[song1], songs_to_song_vectors[song2])
    # song is completely similar with itself
    for song in songs_to_song_vectors.keys():
        similarities[frozenset((song,))] = 1
    return similarities

if __name__ == '__main__':
    pass
