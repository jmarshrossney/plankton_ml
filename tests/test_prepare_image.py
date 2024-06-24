# test_prepare_image.py
import pytest
import torch
from intake_xarray import ImageSource
from cyto_ml.models.scivision import prepare_image

# https://github.com/intake/intake-xarray/blob/d0418f787181d638629b76c2982a9a215a3697be/intake_xarray/image.py#L323


def test_single_image(single_image):

    image_data = ImageSource(single_image).to_dask()
    # Prepare the image
    prepared_image = prepare_image(image_data)

    # Check if the shape is correct (batch dimension added)
    assert prepared_image.shape == torch.Size([1, 3, 80, 79])


def test_image_batch(image_batch):
    """
    Currently expected to fail because dask wants images to share dimensions
    """
    # Load a batch of plankton images

    image_data = ImageSource(image_batch).to_dask()

    with pytest.raises(ValueError) as err:
        prepared_batch = prepare_image(image_data)
        print(err)
    # Check if the shape is correct
    # assert prepared_batch.shape == torch.Size([64, 89, 36, 3])


if __name__ == "__main__":
    pytest.main()
