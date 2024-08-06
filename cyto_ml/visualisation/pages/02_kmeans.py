from sklearn.cluster import KMeans
import streamlit as st
from typing import Optional
from ..visualisation_app import image_embeddings


@st.cache_resource
def kmeans_cluster(n_clusters: Optional[int] = 10):
    """
    K-means cluster the embeddings, option for default size

    """
    X = image_embeddings()
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans


# TODO some visualisation, actual content, etc
