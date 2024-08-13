# Decollages flowcam images (breaks one large tiff into many small ones)
# Attempts to extract coordinate, date and depth information encoded in the filename
# Add add those properties to the resulting output in the EXIF headers
# where file path points to the flowcam data folder which has the collage .tifs and the .lst file inside
# Originally adapted from https://sarigiering.co/posts/extract-individual-particle-images-from-flowcam/
import argparse
import os
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

    meta = lst_metadata(f"{args.filePath}/{lst_file}")

    # create a folder to save the output into
    if os.path.exists(f"{args.filePath}/decollage"):
        pass
    else:
        os.mkdir(f"{args.filePath}/decollage")

    # TODO extract the coords, date, possibly depth from image filename

    # decollage
    # TODO rather than traverse the index and keep rereading large images,
    # filter by filename first and traverse that way, should speed up a lot
    i = 0
    for id in meta["id"]:
        # find collage name and path

        collage_filename = meta["collage_file"][i]
        cp = f"{args.filePath}/{collage_filename}"

        # load collage image
        collage = imread(cp)

        # extract vignette
        #
        img_sub = window_slice(
            cp,
            meta["image_x"][i],
            meta["image_y"][i],
            meta["image_h"][i],
            meta["image_w"][i],
        )

        # TODO write EXIF metadata into the headers, - piexif
        # save vignette to decollage folder
        imsave(f"{args.filePath}/decollage/{args.experimentName}_{id}.tif", img_sub)

        # TODO clean, etc
        i += 1

    # TODO decide whether to do anything with the analytic metadata (circularity etc)
    # We could pop it into a sqlite store at this stage, but want the file linkages
