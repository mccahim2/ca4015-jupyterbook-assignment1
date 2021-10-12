# Introduction


## Iowa Gambling Task (IGT)
This assignment is based around the studies conducted for the [Iowa Gambling Task](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.ak/){cite}'IGT'
In this study 617 healthy participants, the participants have no neurological impairments, are being assessed across 10 independent studies. During the task the partipants are are asked to select on of four cards on a screen, they are given a balance of 2000 USD. The cards they are presented with will either award them or deduct money from their balance. The purpose of the game is to win as much money as possible. In the selection of cards some decks are considered to be "good" (decks 3 and 4) and "bad" (decks 3 and 4), meaning some decks will reward more then others.


## Datasets used
The data being used for this assignment is taken from many different labs ranging in number of participants and number of trials.
:::{note}
The smallest study uses 15 participants while the biggest study uses 162 participants.
:::

The datasets below used various different pay off schemes

The data is gathered from independent studies

Here is a list of the data sets being used for the assignment:

| Study                  | Number of participants | Number of trials |
|------------------------|------------------------|------------------|
| Fridberg et al. 3      | 15                     | 95               |
| Horstmannb             | 162                    | 100              |
| Kjome et al. 5         | 19                     | 100              |
| Maia & McClelland 6    | 40                     | 100              |
| Premkumar et al. 7     | 25                     | 100              |
| Steingroever et al. 8  | 70                     | 100              |
| Steingroever et al. 9  | 57                     | 150              |
| Wetzels et al. 15c     | 41                     | 150              |
| Wood et al. 16         | 153                    | 100              |
| Worthy et al. 17       | 35                     | 100              |

A link to the data is available [here](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.ak/)

Fridberg et al.3, Worth et al., amd Mai&McClelland all fall under payoff scheme no 1.

Horstmann, Steingroever er al8., Steingroever er al9., and Wetzels et al. all fall under payoff scheme no2.

The rest of the studies are parrt of payoff scheme no 3. (Kjome et al., Premkumar et al., and Wood et al.)

The description of the payoff schemes can be found (here)[https://s3-eu-west-1.amazonaws.com/ubiquity-partner-network/up/journal/jopd/ak/sup-text.pdf]

As part of this assignment, k-means clustering will be used on the data sets above.
Clustering is an unsupervised machine learning problem, often used to find interesting patterns in data

K-means clustering is used to assign examples to clusters in an effort to minimize the variance within each cluster.
It is used to quickly predict groupings from within an unlabeled dataset.

Our aim is to model the underlying structure of the data in order to learn from data and identify groups of data (segments/clusters) with similar characteristics/behaviours