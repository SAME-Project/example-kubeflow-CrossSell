import os
import tempfile

from src import download


def test_download():
    with tempfile.TemporaryDirectory() as d:
        download.extract_tar_from_url(data_dir=d)
        assert "data" in os.listdir(d)
        assert "train.csv" in os.listdir(f"{d}/data")
