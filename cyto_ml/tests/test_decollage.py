import pandas as pd
from cyto_ml.data.decollage import lst_metadata


def test_lst_metadata(lst_file):

    df = lst_metadata(lst_file)
    assert isinstance(df, pd.DataFrame)
