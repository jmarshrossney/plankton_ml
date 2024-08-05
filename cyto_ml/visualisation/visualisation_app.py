"""
Streamlit application to visualise how plankton cluster
based on their embeddings from a deep learning model

* Metadata in intake catalogue (basically a dataframe of filenames
  - later this could have lon/lat, date, depth read from Exif headers
* Embeddings in chromadb, linked by filename

"""

import random
from cyto_ml.data.vectorstore import vector_store
import pandas as pd
import requests
from io import BytesIO

from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scivision import load_dataset
from dotenv import load_dotenv
import intake

load_dotenv()


@st.cache_data
def image_ids(collection_name: str) -> list:
    """
    Retrieve image embeddings from chroma database.
    TODO Revisit our available metadata
    """
    collection = vector_store(collection_name)
    result = collection.get()
    return result["ids"]


@st.cache_data
def intake_dataset(catalog_yml) -> intake.catalog.local.YAMLFileCatalog:

    dataset = load_dataset(catalog_yml)
    return dataset


def create_figure(df: pd.DataFrame) -> go.Figure:
    """
    Creates scatter plot based on handed data frame
    TODO replace this layout with
    a) most basic image grid, switch between clusters
    b) ...
    """
    color_dict = {i: px.colors.qualitative.Alphabet[i] for i in range(0, 20)}
    color_dict[-1] = "#ABABAB"
    topic_color = df["topic_number"].map(color_dict)
    fig = go.Figure(
        data=go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker_color=topic_color,
            customdata=df["doc_id"],
            text=df["short_title"],
            hovertemplate="<b>%{text}</b>",
        )
    )
    fig.update_layout(height=600)
    return fig


def main() -> None:
    """
    Main method that sets up the streamlit app and builds the visualisation.
    """
    st.set_page_config(layout="wide", page_title="Plankton image embeddings")
    st.title("Plankton image embeddings")
    col1, col2 = st.columns([3, 1])

    # catalog = "untagged-images-lana/intake.yml"
    # catalog_url = f"{os.environ.get('ENDPOINT')}/{catalog}"
    # ds = intake_dataset(catalog_url)
    # This way we've got a dataframe of the whole catalogue
    # Do we gain even slightly from this when we have the same index in the embeddings
    # index = ds.plankton().to_dask().compute()

    ids = image_ids("plankton")
    test_image_url = random.choice(ids)
    # TODO clean this up
    store = vector_store("plankton")
    # do we even need these
    _ = store.get([test_image_url], include=["embeddings"])["embeddings"]

    # TODO error handling
    response = requests.get(test_image_url)
    st.image(Image.open(BytesIO(response.content)))


if __name__ == "__main__":
    main()
