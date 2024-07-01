"""Try to use the scivision pretrained model and tools against this collection"""

import os
from dotenv import load_dotenv
from cyto_ml.models.scivision import (
    load_model,
    truncate_model,
    prepare_image,
    flat_embeddings,
    SCIVISION_URL,
)
from cyto_ml.data.vectorstore import vector_store
from scivision import load_dataset
from intake_xarray import ImageSource

load_dotenv()


if __name__ == "__main__":

    # Walkthrough here that shows the dataset wrapper being exercised
    # https://github.com/AnnaLinton/scivision_examples/blob/main/how-to-use-scivision.ipynb

    dataset = load_dataset(f"{os.environ.get('ENDPOINT', '')}/metadata/intake.yml")
    collection = vector_store("plankton")

    model = truncate_model(load_model(SCIVISION_URL))

    plankton = (
        dataset.plankton().to_dask().compute()
    )  # this will read a CSV with image locations as a dask dataframe

    # Feels like this is doing dask wrong, compute() should happen later
    # If it doesn't, there are complaints about meta= return value inference
    # that suggest this is wrongheaded use of `apply`: need to learn better patterns
    # So this is a kludge, but we're still very much in prototype territory -
    # Come back and refine this if the next parts work!

    def store_embeddings(row):
        image_data = ImageSource(row.Filename).to_dask()
        embeddings = flat_embeddings(model(prepare_image(image_data)))
        collection.add(
            documents=[row.Filename],
            embeddings=[embeddings],
            ids=[row.Filename],  # must be unique
            # Note - optional arg name is "metadatas" (we don't have any)
        )

    plankton.apply(store_embeddings, axis=1)
