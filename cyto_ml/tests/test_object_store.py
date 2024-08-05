"""Test the layout of the object store is what we expect

untagged-images-lana
untagged-images-wala
tagged-images-lana
tagged-images-wala

Inside tagged-images-lana and tagged-images-wala there is a metadata.csv file and taxonomy.csv file.

"""

import pytest
import s3fs
from cyto_ml.data.s3 import s3_endpoint


def test_endpoint(env_endpoint):
    if not env_endpoint:
        pytest.skip("no settings found for s3 endpoint")

    store = s3_endpoint()
    assert isinstance(store, s3fs.S3FileSystem)


def test_img_ls(env_endpoint):
    if not env_endpoint:
        pytest.skip("no settings found for s3 endpoint")

    store = s3_endpoint()
    for bucket in ["untagged-images-lana", "untagged-images-wala"]:
        filez = store.ls(bucket)
        assert len(filez)
