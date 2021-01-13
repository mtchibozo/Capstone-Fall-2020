# Capstone Fall 2020

## Unsupervised Pattern Recognition from Time Series data

### Contributors
[Romane GOLDMUNTZ](https://github.com/bubblemarchine), [Hritik JAIN](https://github.com/hritik25), [Kassiani Papasotiriou](https://github.com/KassiePapasotiriou), [Amaury SUDRIE](https://github.com/Amelrich), [Maxime TCHIBOZO](https://github.com/mtchibozo), [Vy TRAN](https://github.com/vttran4)

### Problem Statement

In financial trading, technical analysis aims to evaluate investments and identify opportunities using only a stock’s price and volume data. Under this assumption, predictions are made by identifying patterns which are empirically “known” to lead to a predetermined outcome. 

For example, the “head and shoulders” pattern is a reversal sign, indicating a change from bull to bear trend. However, the presence of multiple patterns in a given time series adds complexity to a technical analyst’s task of predicting stock behaviors. A pattern may be 70% head and shoulders, and 20% channel up, for example.

In this capstone project, we sought to best utilize our backgrounds in data science to approach technical analysis in an objective and a data-driven way.

The primary question we aim to answer is, *can we identify meaningful patterns in stock data using unsupervised learning, given the multiscale nature of these financial time series?*

### Methodology Overview

We break the multiscale pattern detection problem down into three subproblems.

* First, how can we gather data which has multiscale properties? We use real data from the S&P 500 Index over the past twenty years, and also generate artificial data to help us test our algorithms. 

* Secondly, how do we extract multiscale features from this data? To this end, we explore a variety of methods from financial engineering, physics, and machine learning. More specifically:

| Preprocessing Methods               |
|-------------------------------------|
| Discrete Cosine Transforms          |
| Fourier Transform                   |
| Autoencoders                        |
| Perceptually Important Points (PIP) |
| Padded & Skipped Sampling           |

We cluster the features extracted through these methods using a KMeans algorithm and evaluate the clustering results with appropriate metrics. We then combine the methods that are found to produce high quality clusters to obtain complete clustering pipelines. 

* Finally, to verify that our methods are capturing multiscale patterns, we look into the PIP Permutation Entropy in the resulting clusters, and the Shannon Entropy of Conditional Probability Distributions of independently clustered long and short time series.


### Results

##### Clustering Results

For our clustering evaluation we will present results based on combining the Silhouette Score with the Elbow Method.

| Preprocessing Method             | k* | Average S*-score for k* |
|----------------------------------|----|-------------------------|
| DCT - Autoencoders               | 7  | 0.24                    |
| DCT - PIP Embedding Only         | 6  | 0.164                   |
| DCT-Skipped Values               | 6  | 0.132                   |
| DCT-Padded Values                | 6  | 0.070                   |
| Fourier Transform-Skipped Values | 6  | 0.146                   |
| Fourier Transform-Padded Values  | 6  | 0.146                   |

#### Multiscale evaluation

To evaluate whether our clusters capture multiscale we used PIP Permutation entropy and Conditional Probability & Shannon entropy. More information and the complete set of results can be found here (to be updated).

### Code

All the preprocessing methods and multiscale evaluation notebooks can be found in their respective folder

### Publications

(to be updated)



