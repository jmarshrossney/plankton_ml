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
from typing import Optional
import intake

load_dotenv()

STORE = vector_store("plankton")


@st.cache_data
def image_ids(collection_name: str) -> list:
    """
    Retrieve image embeddings from chroma database.
    TODO Revisit our available metadata
    """
    result = STORE.get()
    return result["ids"]


@st.cache_data
def intake_dataset(catalog_yml: str) -> intake.catalog.local.YAMLFileCatalog:

    dataset = load_dataset(catalog_yml)
    return dataset


def closest_n(url: str, n: Optional[int] = 26) -> list:
    embed = STORE.get([url], include=["embeddings"])["embeddings"]
    results = STORE.query(query_embeddings=embed, n_results=n)
    return results["ids"][0]  # by index because API assumes query always multiple


def closest_grid(start_url: str, rows: list):
    closest = closest_n(start_url)
    # TODO error handling

    for index, r in enumerate(rows):
        for c in rows[index]:
            # TODO cache for this
            response = requests.get(closest.pop())

            c.image(Image.open(BytesIO(response.content)))


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
    # it starts much slower on adding this
    # the generated HTML is not lovely at all
    rows = []
    for i in range(0, 5):
        rows.append(st.columns(5))

    # catalog = "untagged-images-lana/intake.yml"
    # catalog_url = f"{os.environ.get('ENDPOINT')}/{catalog}"
    # ds = intake_dataset(catalog_url)
    # This way we've got a dataframe of the whole catalogue
    # Do we gain even slightly from this when we have the same index in the embeddings
    # index = ds.plankton().to_dask().compute()

    ids = image_ids("plankton")
    # starting image

    test_image_url = random.choice(ids)

    # TODO figure out how streamlit is supposed to work
    closest_grid(test_image_url, rows)


if __name__ == "__main__":
    main()
