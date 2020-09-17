import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

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