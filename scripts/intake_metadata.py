"""Convert the metadata into format usable with `intake`,
for trial use with `scivision`:
https://scivision.readthedocs.io/en/latest/api.html#scivision.io.reader.load_dataset
https://intake.readthedocs.io/en/latest/catalog.html#yaml-format

See also https://github.com/intake/intake-stac
Via https://gallery.pangeo.io/repos/pangeo-data/pangeo-tutorial-gallery/intake.html#Build-an-intake-catalog

"""

from cyto_ml.data.intake import intake_yaml
from cyto_ml.data.s3 import s3_endpoint
import pandas as pd
import os


def load_metadata(path: str):
    return pd.read_csv(f"{os.environ['ENDPOINT']}/{path}")


if __name__ == "__main__":
    metadata = load_metadata("metadata/metadata.csv")

    # rewrite it to add the full s3 image path
    metadata["Filename"] = metadata["Filename"].apply(
        lambda x: f"{os.environ['ENDPOINT']}/untagged-images/{x}"
    )

    fs = s3_endpoint()
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

        out.write(intake_yaml(cat_test, cat_url))
