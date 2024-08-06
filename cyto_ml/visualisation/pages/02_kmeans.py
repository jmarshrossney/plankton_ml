from sklearn.cluster import KMeans
import streamlit as st
from cyto_ml.visualisation.visualisation_app import (
    image_embeddings,
    image_ids,
    cached_image,
)


@st.cache_resource
def kmeans_cluster() -> KMeans:
    """
    K-means cluster the embeddings, option for default size

    """
    print("model")
    X = image_embeddings("plankton")
    n_clusters = st.session_state["n_clusters"]
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans


@st.cache_data
def image_labels() -> dict:
    """
    TODO good form to move all this into cyto_ml, call from there
    """
    km = kmeans_cluster()
    clusters = dict(zip(set(km.labels_), [[] for _ in range(len(set(km.labels_)))]))

    for index, id in enumerate(image_ids("plankton")):
        label = km.labels_[index]
        clusters[label].append(id)
    return clusters


def show_cluster():

    # TODO n_clusters configurable with selector
    fitted = image_labels()
    closest = fitted[st.session_state["cluster"]]

    # seems backwards, something in session state?
    rows = []
    for _ in range(0, 8):
        rows.append(st.columns(8))
    for index, _ in enumerate(rows):
        for c in rows[index]:
            c.image(cached_image(closest.pop()), width=60)


# TODO some visualisation, actual content, etc
def main() -> None:

    if "cluster" not in st.session_state:
        st.session_state["cluster"] = 1
    if "n_clusters" not in st.session_state:
        st.session_state["n_clusters"] = 5

    st.selectbox(
        "cluster",
        [x for x in range(0, st.session_state["n_clusters"])],
        key="cluster",
        on_change=show_cluster,
    )

    show_cluster()


if __name__ == "__main__":
    main()
