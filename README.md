## **Genre Classification**

I've chosen to build a genre classifier! This project has the most readily available data (scraping Spotify using the `spotipy` library).

## **Problem Statement**

Using features derived from an audio source itself, can a classification model predict the genre of a 30-second audio clip with high enough accuracy to organize new songs into their respective genres?

Classifying genre is important for music distribution and streaming platforms, it also helps listeners find new bands they might like, and it, in turn, helps musicians connect with new audiences.

My goal is to be able to predict genre using **features derived from the audio signal** to classify genre. This could be helpful for building playlists algorithmically without needing humans to manually input data, and can help artists assign their music to all the genres that their music fits the description of.

**The metric for success here is accuracy**--any incorrect response is a bad one.

## **Methods and models**

### Data Collection

Scraping 30-second song samples using `spotipy` - I will start with 5 general genres and dig down into the subgenres of each:

* Original 5 Genres:
  * Classical
  * Progressive Bluegrass
  * Rock
  * Rap
  * R & B
 
* Extra 5 Genres:
 * Tropical House
 * Pop
 * Baroque
 * Serialism
 * Hip Hop
 
Note that after my first round of modeling, I was getting about 84% accuracy with the original 5 genres. With classical, **I was getting 95% accuracy.** I included Baroque, Serialism, and Ambient in the additional set of 5 genres to test the suspicion that Classical was easier to predict **because it was quieter** than the other 4 original genres. Baroque and Serialism are subsets of classical music. The **Baroque period is from approximatlely the 1600s-1800s** (there's debate about the blurry start and end, but it's not particularly important here). **Serialism is a subset of classical music emerging in the 20th century**. If my model can continue to predict between classical, baroque, and serialism well given these closely-related genres, I'd be very impressed!

Similarly, I included Tropical House and Folk to try to add genres closely related to the Rap and Progressive Bluegrass genres, respectively. Unfortunately, Folk didn't have enough unique songs to meet my threshold of at least 500 unique songs before sampling.

### Methods for cleaning and preprocessing data

Using mostly the `librosa` library along with existing techniques gleaned from the [Music information retrieval](https://musicinformationretrieval.com/index.html) community, I'll use various methods for extracting features from the music itself. The methods I use to extract features from the audio are:
* `Energy and Root Mean Squared Energy (RMSE)` - energy and RMSE are different measurements of loudness, can be measured in windows across a 30-sec range
* `Fast Fourier Transform (fft)` - converting a time-window of an audio source into a snapshot of the frequency spectrum
* `Mel Frequency Cepstral Coefficients (mfcc)` - Creates overlapping bins along the log frequency spectrum and stores the power of each of those bins across windows of time.

### Models

**Support Vector Machines** and **Convolutional Neural Networks** seem to get good results when dealing with audio classification problems. Audio signal can be converted into images using the above preprocessing steps.[1]

## **Results**

My best performing models were `SVC` models with varying C values tuned to minimize overfitting

|     Spotify features |     My Models    |     Number of Genres    |
|----------------------|------------------|-------------------------|
|     78.4%            |     84.2%        |     5                   |
|     50.6%            |     56.5%        |     10                  |

#### Percentage of incorrect classifications per genre using **Spotify's features** on 5 genres

| Genre                     | Percent_of_wrong_predictions |
|---------------------------|------------------------------|
| rap                       | 31.94                        |
| r&b                       | 26.39                        |
| rock                      | 20.14                        |
| progressive bluegrass     | 13.19                        |
| classical                 | 8.33                         |

#### Percentage of incorrect classification per genre using **my extracted features** on 5 genres

| Genre                 | pct_predicted_wrong |
|-----------------------|---------------------|
| r&b                   | 33.33               |
| rap                   | 25.49               |
| progressive bluegrass | 18.62               |
| rock                  | 16.66               |
| classical             | 5.88                |

#### Percentage of incorrect classifications per genre using **Spotify's features** on 10 genres

| Genre                 | pct_predicted_wrong |
|-----------------------|---------------------|
| pop                   | 17.31               |
| classical             | 15.53               |
| hip hop               | 14.23               |
| r&b                   | 13.10               |
| rap                   | 11.32               |
| baroque               | 7.76                |
| rock                  | 6.63                |
| tropical house        | 4.85                |
| serialism             | 4.69                |
| progressive bluegrass | 4.53                |

#### Percentage of incorrect classifications per genre using **my extracted features** on 10 genres

| Genre                 | Pct_predicted_wrong |
|-----------------------|---------------------|
| pop                   | 18.93               |
| hip hop               | 15.99               |
| classical             | 15.99               |
| r&b                   | 11.02               |
| rap                   | 11.02               |
| baroque               | 8.82                |
| serialism             | 5.69                |
| tropical house        | 4.77                |
| rock                  | 4.04                |
| progressive bluegrass | 3.67                |

## **Conclusions**

* My models using extracted MFCCs consistently outperform models using spotify's features.
* When subgenres are present, models tend to predict related genres. For example: classical, baroque, and serialism were the most likely to be confused for each other, but those all fit under the genre umbrella of classical
* Further exploration needs to be done to figure out why the subgenres are being classified together - OR this could be an interesting way to find intersections between genres that are sonically adjacent to each other.