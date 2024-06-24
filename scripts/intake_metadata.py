"""Convert the metadata into format usable with `intake`,
for trial use with `scivision`:
https://scivision.readthedocs.io/en/latest/api.html#scivision.io.reader.load_dataset
https://intake.readthedocs.io/en/latest/catalog.html#yaml-format

See also https://github.com/intake/intake-stac 
Via https://gallery.pangeo.io/repos/pangeo-data/pangeo-tutorial-gallery/intake.html#Build-an-intake-catalog

"""

import s3fs
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()


def load_metadata(path: str):
    return pd.read_csv(f"{os.environ['ENDPOINT']}/{path}")


def write_yaml(test_url: str, catalog_url: str, ):
    """
    Write a minimal YAML template describing this as an intake datasource
    Example: plankton dataset made available through scivision, metadata
    https://raw.githubusercontent.com/alan-turing-institute/plankton-cefas-scivision/test_data_catalog/scivision.yml
    See the comments below for decisions about its structure
    """
    template = f"""
sources:
  test_image:
    description: Single test image from the plankton collection
    origin: 
    driver: intake_xarray.image.ImageSource
    args:
      urlpath: ["{test_url}"]
      exif_tags: False
  plankton:
    description: A CSV index of all the images of plankton
    origin: 
    driver: intake.source.csv.CSVSource
    args:
      urlpath: ["{catalog_url}"]
"""
    # coerce_shape: [256, 256]
    return template


if __name__ == "__main__":
    metadata = load_metadata("metadata/metadata.csv")

    # rewrite it to add the full s3 image path
    metadata["Filename"] = metadata["Filename"].apply(
        lambda x: f"{os.environ['ENDPOINT']}/untagged-images/{x}"
    )

    # may not need this unless we choose to write back for completeness
    fs = s3fs.S3FileSystem(
        anon=False,
        key=os.environ.get("FSSPEC_S3_KEY", ""),
        secret=os.environ.get("FSSPEC_S3_SECRET", ""),
        client_kwargs={"endpoint_url": os.environ["ENDPOINT"]},
    )

    # Option to use a CSV as an index, rather than return the files
    catalog = "metadata/catalog.csv"
    with fs.open(catalog, "w") as out:
        out.write(metadata.to_csv())
    cat_url = f"{os.environ['ENDPOINT']}/{catalog}"

    with fs.open("metadata/intake.yml", "w") as out:
        # Do we use a CSV driver and include the metadata?
        # out.write(write_yaml(f"{os.environ['ENDPOINT']}/{catalog}"))

        # All the scivision examples have image collections in a single zipfile
        # This format throws an s3 error on the directory listing - 
        # unsure if this is a permissions issue, or you just can't use a wildcard
        cat_wildcard = f"{os.environ['ENDPOINT']}/untagged-images/*.tif"  # .replace('https://', 's3://')

        # Create a testing record for a single file
        cat_test = cat_wildcard.replace("*", "19_10_Tank22_1")

        # Our options for the whole collection look like:
        # * a tiny http server that creates a zip, but assumes the images have more metadata
        # * a tabular index instead, means we get less advantage from intake though

        out.write(write_yaml(cat_test, cat_url))
