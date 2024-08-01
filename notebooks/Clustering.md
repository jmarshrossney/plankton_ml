---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: cyto_39
    language: python
    name: python3
---

Notebook examination of applying unsupervised clustering methods to vector embeddings and visualising the results

Keywords: K-means, DBScan, T-SNE, other?

Paper reference for approach: https://aslopubs.onlinelibrary.wiley.com/doi/full/10.1002/lno.12101#lno12101-sec-0025-title

Possibly interesting if we try the transformer-based plankton model from Turing: https://link.springer.com/chapter/10.1007/978-3-030-74251-5_23 


```python
import sys
sys.path.append('../')
from cyto_ml.data.vectorstore import vector_store, client
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
```

Load our embeddings into a form suitable for throwing at clustering algorithms, 2048 features might be optimistic and we need to first reduce them!

```python
store = vector_store('plankton')
res = store.get(include=['embeddings'])
X = res['embeddings']
```

This doesn't work with such a high number of features even with `make_blobs` generating pre-clustered data with 2048 features, and tips like scaling values.

So either PCA first, or we just stick with K-means as a simpler effort and work our way back here.

```python
def do_dbscan(X):
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    db = DBSCAN(eps=0.7, min_samples=100).fit(X_scaled)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

do_dbscan(X)
```

Sense check on generated dataset with same number of features and three natural clusters - uncomment to see it

```python
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, centers=3, n_features=2048,
                  random_state=0)
# do_dbscan(X)
```

Fall back to a K-means approach, just to try and get some visual feedback

```python
# Set the number of clusters
num_clusters = 10  # Adjust based on your data

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_
```

```python
len(labels)
```

```python

clusters = dict(zip(set(labels), [[] for _ in range(len(set(labels)))]))

for index, id in enumerate(res['ids']):
    l = labels[index]
    clusters[l].append(id)
```

```python
i = 3 # picked at random
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(10., 10.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                 axes_pad=0.2,  # pad between axes in inch.
                 )

for index, ax in enumerate(grid):
    # Iterating over the grid returns the Axes.
    ax.imshow(io.imread(clusters[i][index]))

```

To be continued

* Iteration with cluster sizes - 10 was picked arbitrarily, 1 and 2 look like detritus
* Proper look at image quality - what's getting lost between the FlowCam and here
* Nicer way of doing this than a notebook, that has some level of reuse value for other image projects

