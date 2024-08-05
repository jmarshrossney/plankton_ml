"""
Streamlit application to visualise how plankton cluster
based on their embeddings from a deep learning model

* Metadata in intake catalogue (basically a dataframe of filenames
  - later this could have lon/lat, date, depth read from Exif headers
* Embeddings in chromadb, linked by filename

"""

import chromadb
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


CDB = None


def get_chroma_client() -> chromadb.Client:
    """
    Retrieve or instantiate the chromadb client.
    """
    global CDB
    if CDB is None:
        CDB = chromadb.HttpClient(host="localhost", port=8000)
    return CDB


@st.cache_data
def get_embeddings(collection_name: str) -> list:
    """
    Retrieve image embeddings from chroma database.
    TODO Revisit our available metadata
    """
    collection = get_chroma_client().get_collection(collection_name)
    result = collection.get(include=["embeddings"])
    return result["embeddings"]


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

    # test_image =
    # st.image(


if __name__ == "__main__":
    main()
