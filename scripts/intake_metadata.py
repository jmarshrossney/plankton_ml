"""Heavy-handed approach to create image metadata in usable with `intake`,
for trial use with `scivision`:
https://scivision.readthedocs.io/en/latest/api.html#scivision.io.reader.load_dataset
https://intake.readthedocs.io/en/latest/catalog.html#yaml-format

See also https://github.com/intake/intake-stac
Via https://gallery.pangeo.io/repos/pangeo-data/pangeo-tutorial-gallery/intake.html#Build-an-intake-catalog

"""

import os
from cyto_ml.data.intake import intake_yaml
from cyto_ml.data.s3 import s3_endpoint, image_index


if __name__ == "__main__":

    fs = s3_endpoint()

    # TODO this is a minimal change to only reflect the Lancaster data
    # Need looking harder at the Wallingford data to decide how to treat it
    # They're really distinct datasets, any benefit to sharing an index?
    image_bucket = "untagged-images-lana"

    metadata = image_index(fs, image_bucket)

    # Option to use a CSV as an index, rather than return the files
    catalog = f"{image_bucket}/catalog.csv"
    with fs.open(catalog, "w") as out:
        out.write(metadata.to_csv(index=False))

    cat_url = f"{os.environ['ENDPOINT']}/{catalog}"

    with fs.open(f"{image_bucket}/intake.yml", "w") as out:
        # Do we use a CSV driver and include the metadata?
        # out.write(write_yaml(f"{os.environ['ENDPOINT']}/{catalog}"))

        # See the issue here: https://github.com/NERC-CEH/plankton_ml/issues/3
        # About data improvements needed before a better way to read a bucket into s3
        cat_test = f"{os.environ['ENDPOINT']}/untagged-images-lana/19_10_Tank22_1.tif"

        # Options for the whole collection look like:
        # 1. a tiny http server that creates a zip, but assumes the images have more metadata
        # 2. a tabular index instead, means we get less advantage from intake though
        # We've gone with 2, needs documentation and continual revisiting
        out.write(intake_yaml(cat_test, cat_url))
