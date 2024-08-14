# Decollages flowcam images (breaks one large tiff into many small ones)
# Attempts to extract coordinate, date and depth information encoded in the filename
# Add add those properties to the resulting output in the EXIF headers
# where file path points to the flowcam data folder which has the collage .tifs and the .lst file inside
# Originally adapted from https://sarigiering.co/posts/extract-individual-particle-images-from-flowcam/
import argparse
import os
import re
import pandas as pd
import numpy as np
from skimage.io import imread, imsave


def lst_metadata(filename: str) -> pd.DataFrame:
    """
    Read the csv-ish ".lst" file from the FlowCam export
    Return a pandas dataframe
    """
    heads = pd.read_csv(filename, sep="|", nrows=53, skiprows=1)
    colNames = list(heads["num-fields"])
    meta = pd.read_csv(filename, sep="|", skiprows=55, header=None)
    meta.columns = colNames
    return meta


def window_slice(
    image: np.ndarray, x: int, y: int, height: int, width: int
) -> np.ndarray:
    return image[y : y + height, x : x + width]  # noqa: E203


def headers_from_filename(filename: str) -> dict:
    """Attempt to extract lon/lat and date, option of depth, from filename
    Return a dict with key-value pairs for use as EXIF headers
    """
    headers = {}
    pattern = r"_(-?\d+\.\d+)_(-?\d+\.\d+)_(\d{8})(?:_(\d+))?"

    match = re.search(pattern, filename)
    if match:
        lat, lon, date, depth = match.groups()
        # https://exiftool.org/TagNames/GPS.html
        headers["GPSLatitude"] = lat
        headers["GPSLongitude"] = lon
        headers["DateTimeOriginal"] = (
            date  # better to leave as date than pad with zero hours?
        )
        # TODO most depth matches will be spurious, what are the rules (refer to Kelly?
        headers["GPSAltitude"] = (
            depth  # can we use negative altitude as bathymetric depth?
        )
    return headers


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="FlowCam_DeCollager",
        description="Decollages flow cam images. requires pandas (pip install pandas) and cv2 (pip install opencv-python).",  # noqa: E501
    )
    parser.add_argument(
        "filePath",
        help="path to the flowcam data file which contains collage .tifs and an .lst file",
    )
    parser.add_argument("experimentName", help="name to append to each decollaged file")
    args = parser.parse_args()

    files = os.listdir(args.filePath)

    # TODO handle cleaner
    lst_file = False
    for f in files:
        if f.endswith(".lst"):
            lst_file = f
            break

    if not lst_file:
        raise FileNotFoundError("no lst file in this directory")

    metadata = lst_metadata(f"{args.filePath}/{lst_file}")

    # TODO consider squirting this straight into the object store API
    # create a folder to save the output into
    if os.path.exists(f"{args.filePath}/decollage"):
        pass
    else:
        os.mkdir(f"{args.filePath}/decollage")

    # TODO extract the coords, date, possibly depth from image filename

    # decollage - rather than traverse the index and keep rereading large images,
    # filter by filename first and traverse that way, should speed up a lot
    for collage_file in metadata.groupby.unique():

        # Read lat, lon etc
        collage_headers = headers_from_filename(collage_file)

        collage = imread(f"{args.filePath}/{collage_file}")

        df = metadata[metadata.collage_file == collage_file]

        for i in df.index:
            # extract vignette
            height = df["image_h"][i]
            width = df["image_w"][i]
            img_sub = window_slice(
                collage,
                df["image_x"][i],
                df["image_y"][i],
                height,
                width,
            )
            # TODO write EXIF metadata into the headers, - piexif or exiftool?
            # https://www.exiftool.org/faq.html#Q14
            headers = collage_headers
            headers["ImageWidth"] = width
            headers["ImageHeight"] = height
            # save vignette to decollage folder
            imsave(f"{args.filePath}/decollage/{args.experimentName}_{id}.tif", img_sub)

    # TODO decide whether to do anything with the analytic metadata (circularity etc)
    # We could pop it into a sqlite store at this stage, but want the file linkages
