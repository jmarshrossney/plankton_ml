import pandas as pd
from skimage.io import imread
from cyto_ml.data.decollage import lst_metadata, window_slice, headers_from_filename


def test_lst_metadata(lst_file):
    df = lst_metadata(lst_file)
    assert isinstance(df, pd.DataFrame)


def test_window_slice(collage_file):
    img = imread(collage_file)
    win = window_slice(img, 5, 5, 25, 50)
    assert win.shape == (25, 50, 3)


def test_headers_from_filename(collage_file):
    h = headers_from_filename(collage_file)
    assert "GPSLatitude" in h and h["GPSLatitude"]
