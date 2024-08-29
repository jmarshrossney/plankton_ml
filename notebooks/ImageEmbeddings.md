---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

Use this with the `cyto_ml` environment.

`conda env create -f environment.yml`
`conda activate cyto_ml`

```python
import os
from dotenv import load_dotenv
import torch
import torchvision
import chromadb
import sys
sys.path.append('../')
from cyto_ml.models.scivision import prepare_image
from intake_xarray import ImageSource
load_dotenv()  # sets our object store endpoint and credentials from the .env file

from intake import open_catalog

from resnet50_cefas import load_model
```

```python
dataset = open_catalog(f"{os.environ.get('ENDPOINT', '')}/metadata/intake.yml")
model = load_model()
dataset.test_image().to_dask()
```

```python
network = load_model(strip_final_layer=True)
```

```python
imgs = dataset.test_image().to_dask()
imgs.to_numpy().shape
```

Pass the image through our truncated network and get some embeddings out

```python
o = prepare_image(imgs)
feats = network(o)
feats.shape
```

```python
embeddings = feats[0].tolist()
```

```python
embeddings
```

```python
print(set([type(x) for x in embeddings]))
```

There's been some guesswork up to this point (honestly expected the input would be resized, and concerned it's not normalised) but we have a set of embeddings, at which point pop them in a vector database and see if similarity queries yield any kind of meaningful results, iterate from there

https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/ - langchain has a few, Milvus would be another option
https://python.langchain.com/v0.1/docs/integrations/vectorstores/ - there are so many integrations!
https://www.trychroma.com/ - this looks straightforward to approach

```python
client = chromadb.PersistentClient(path="./vectors")
collection = client.create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"} # l2 is the default
    )
```

```python
collection.add(
    documents=["test_image"],
    embeddings=[embeddings],
    metadatas=[{"useful": "maybe"}],
    ids=["id2"]  # must be unique, are they required?
)
```

```python
collection.get('id2',include=["embeddings"])
```

```python
index = dataset.plankton().to_dask().compute()
```

```python
index

```

```python
def flat_embeddings(features: torch.Tensor):
    return features[0].tolist()
```

```python
def file_embeddings(row):
    image_data = ImageSource(row.Filename).to_dask()
    embeddings = flat_embeddings(network(prepare_image(image_data)))
    collection.add(
        documents=[row.Filename],
        embeddings=[embeddings],
        ids=[row.Filename]  # must be unique, are they required?
    )


```

```python
from intake_xarray import ImageSource
```

Because all the images have slightly different dimensions as they come out of the FlowCam, we can't batch them
Push them through the model one by one and either build a list of `(id, [embeddings])` pairs, or potentially pop them straight into chromadb as we apply the function, which would keep it more dasklike?

This scales ok at 8000 or so images

```python
collection.count()
```

This is _really_ slow - joe

```python
res = index.apply(file_embeddings, axis=1)
```

```python
collection.count()
```

```python
i = ImageSource(index.loc[0].Filename).to_dask()
i
```

```python
network(prepare_image(i))
```
