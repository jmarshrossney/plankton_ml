"""Try to use the scivision pretrained model and tools against this collection"""

import os
from dotenv import load_dotenv
from cyto_ml.models.scivision import load_model, truncate_model, prepare_image, SCIVISION_URL
from cyto_ml.data.vectorstore import vector_store
from scivision import load_dataset

load_dotenv()


if __name__ == "__main__":

    # Walkthrough here that shows the dataset wrapper being exercised
    # https://github.com/AnnaLinton/scivision_examples/blob/main/how-to-use-scivision.ipynb

    dataset = load_dataset(f"{os.environ.get('ENDPOINT', '')}/metadata/intake.yml")

    imgs = dataset.test_image().to_dask() # this will read a single image as an xarray

    vecs = vector_store()

    model = truncate_model(load_model(SCIVISION_URL))

    embeddings = model(prepare_image(imgs))

    print(embeddings)

    plankton = dataset.plankton().to_dask() # this will read a CSV with image locations as a dask dataframe
 
