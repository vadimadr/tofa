from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).parent
SOURCE_ROOT = TEST_ROOT.parent
TEST_DATA = TEST_ROOT.joinpath("test_data")


class GetTestAssetFixture:
    def __init__(self, asset_path):
        self.asset_path = asset_path

    def __call__(self, asset_name):
        return self.asset_path / asset_name


@pytest.fixture
def get_asset():
    return GetTestAssetFixture(TEST_DATA)
