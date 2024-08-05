"""
Streamlit application to view EIDC datasets using their document embeddings
"""

from ast import literal_eval

import chromadb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Union  # lose the 3.10+ type syntax with |, mixed feelings

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
def get_embeddings(collection_name: str) -> pd.DataFrame:
    """
    Retrieve document embeddings from chroma database.
    TODO replace this with image embeddings, revisit our available metadata
    """
    collection = get_chroma_client().get_collection(collection_name)
    result = collection.get(include=["metadatas"])
    reduced_embeddings = [
        literal_eval(metadata["umap_reduced"]) for metadata in result["metadatas"]
    ]
    df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    df["title"] = [metadata["title"] for metadata in result["metadatas"]]
    df["description"] = [metadata["description"] for metadata in result["metadatas"]]
    df["lineage"] = [metadata["lineage"] for metadata in result["metadatas"]]
    df["topic"] = [metadata["topic_keywords"] for metadata in result["metadatas"]]
    df["topic_number"] = [metadata["topic_number"] for metadata in result["metadatas"]]
    df["doc_id"] = result["ids"]
    df["short_title"] = [
        title[:50] + "..." if len(title) > 15 else title
        for title in df["title"].to_list()
    ]
    return df


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


def update_text(title: str, desc: str, topic: str, col) -> None:
    """
    Updates the texts in the passed column with details of the currently
    selected dataset.
    """
    with col:
        st.markdown(f"**{title}**")
        st.markdown(f"*{topic}*")
        st.markdown(desc)


def extract_details(df: pd.DataFrame, doc_id: str) -> Union[str, str, str]:
    """
    Extract title, description and topic details from a dataframe based on
    the an id.
    """
    selection = df[df["doc_id"] == doc_id]
    title = selection["title"].iloc[0]
    desc = selection["description"].iloc[0]
    topic = selection["topic"].iloc[0]
    return title, desc, topic


def main() -> None:
    """
    Main method that sets up the streamlit app and builds the visualisation.
    """
    st.set_page_config(layout="wide", page_title="Plankton image embeddings")
    st.title("Plankton image embeddings")
    col1, col2 = st.columns([3, 1])

    df = get_embeddings("plankton")
    fig = create_figure(df)

    event = col1.plotly_chart(
        fig, key="embeddings", on_select="rerun", selection_mode="points"
    )
    if len(event["selection"]["points"]) > 0:
        point = event.selection.points[0]
        doc_id = point["customdata"]
        title, desc, topic = extract_details(df, doc_id)
        update_text(title, desc, topic, col2)


if __name__ == "__main__":
    main()
