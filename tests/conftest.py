import os
import pytest

from resnet50_cefas import load_model


@pytest.fixture
def image_dir():
    """
    Existing directory of images
    """
    return os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "fixtures/test_images/"
    )


@pytest.fixture
def single_image(image_dir):
    # The file naming conventions were like this when i got here
    return os.path.join(image_dir, "testymctestface_36.tif")


@pytest.fixture
def image_batch(image_dir):
    return os.path.join(image_dir, "testymctestface_*.tif")


@pytest.fixture
def scivision_model():
    return load_model(strip_final_layer=True)


@pytest.fixture
def env_endpoint():
    """None if ENDPOINT is not set in environment,
    or it's set but to an arbitrary string,
    utility for skipping integration-type tests"""
    endpoint = os.environ.get("ENDPOINT", None)
    # case in which we've got blether in the default config
    if endpoint and "https" not in endpoint:
        endpoint = None
    return endpoint
