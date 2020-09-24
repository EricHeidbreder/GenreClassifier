import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
import time
import spotipy
import urllib
from tqdm import tqdm
from auth import generate_token

def plot_signals(signals):
    nrows = int(len(signals) / 5)
    ncols = int(len(signals) / 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                            sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(nrows):
        for y in range(ncols):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            
def plot_fft(fft, signals):
    nrows = int(len(signals) / 5)
    ncols = int(len(signals) / 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                            sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(nrows):
        for y in range(ncols):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank, signals):
    nrows = int(len(signals) / 5)
    ncols = int(len(signals) / 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                            sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(nrows):
        for y in range(ncols):
            axes[x, y].set_title(list(fbank.keys())[i])
            axes[x, y].imshow(list(fbank.values())[i],
                             cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1
            
def plot_mfccs(mfccs, signals):
    nrows = int(len(signals) / 5)
    ncols = int(len(signals) / 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                            sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(nrows):
        for y in range(ncols):
            axes[x, y].set_title(list(mfccs.keys())[i])
            axes[x, y].imshow(list(mfccs.values())[i],
                             cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1
            
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs) # since signal goes above and below x-axis
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask
            
def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1 / rate)
    Y = abs(np.fft.rfft(y) / n)
    return (Y, freq)

def playlist_to_genres(user_id, playlist_id):
    '''
    Function to iterate through a playlist, pull 30-second samples of songs,
    and separate them out into their genres. Some songs have multiple
    genres, so they will appear in multiple folders.
    '''
    token = generate_token()
    spotify = spotipy.Spotify(auth=token) # Authorization token
    results = spotify.user_playlist(user=user_id, 
                                    playlist_id=playlist_id)

    # Getting track, url, and artists on a track
    tracks = results['tracks']
    
    # Setting cols for DataFrame
    track_info_cols = ['id', 'genre', 'track_name', 'preview_url', 'location', 'filename', 'artist', 'artist_uri', 'album', 'release_date', # Track info
                      'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'] # Audio features
    
    # Setting a timestamp for .csv versioning
    timestamp = time.ctime().replace(' ', '_').replace(':', '_')
    
    # Set filepath name
    filepath = f'./data/song_genres.csv'
    
    # -------------MAIN LOOP-------------
    while True:
        # We need to create a dataframe to store the results, going to store the 
        # results at each stage in case the process gets interrupted
        song_df = pd.DataFrame(columns=track_info_cols)
        
        for track in tqdm(tracks['items']):
            track_info = track['track']
            # Some songs don't have ids, apparently?
            try:
                track_id = track_info['id']
            except:
                continue
            # Some songs don't have preview_urls
            try:
                url = track_info['preview_url'] # 30-sec song sample
            except:
                continue
            artists = track_info['artists']
            album_name = track_info['album']['name']
            release_date = track_info['album']['release_date']
            # This returns the audio features Spotify has created for each song
            audio_features = spotify.audio_features(track_info['uri'])[0] 

            # Looping through each artist, getting their name and uri (Spotify's unique identifier)  
            for artist in artists:
                artist_name = artist['name']
                artist_uri = artist['uri']
                track_name = track_info['name']
                try:
                    genres = spotify.artist(artist_uri)['genres']
                except:
                    continue

                # urls would throw AttributeError occasionally, some songs don't have previews
                try:
                    # There are multiple genres per artist
                    for genre in genres:
                        # Now let's get the 30-second song sample into something we can use
    #                     url = track_info['preview_url']
#                         mp3file = urllib.request.urlopen(url)
#                         os.makedirs(f'./scrapes/{genre}', exist_ok=True)

    #                     # Let's write this song to a folder!
    #                     with open(f'./scrapes/{genre}/{track_name.strip(punctuation)}.mp3','wb') as output:
    #                           output.write(mp3file.read())

                        # Add info to dictionary
                        row = {'id' : track_id,
                               'genre' : genre,
                              'track_name' : track_name,
                              'preview_url' : url,
                              'location' : f'./scrapes/{genre}/{track_name}.mp3',
                              'filename' : f'{track_name}.mp3',
                              'artist' : artist_name,
                              'artist_uri' : artist_uri,
                              'album' : album_name,
                              'release_date' : release_date 
                              }
                        # Need to create a dictionary for the audio features
                        audio_feature_dict = {key: value for key, value in audio_features.items() if key in track_info_cols}
                        
                        # Combine the above row and audio_feature_dict into one
                        row.update(audio_feature_dict)

                        song_df = song_df.append(row, ignore_index=True)

                except AttributeError:
                    pass
                
        if tracks['next']:
            tracks = spotify.next(tracks) # updates the offset by 100            
            song_df.to_csv(filepath,
                          mode='a',
                          index=False)
        else:
            break
            
    song_df.to_csv(filepath, 
                   mode='a', 
                   index=False)