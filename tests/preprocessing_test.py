import os
import pathlib
import tempfile

from shutil import copyfile

from src import preprocessing


CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


def test_preprocessing():
    with tempfile.TemporaryDirectory() as out_d:
        with tempfile.TemporaryDirectory() as in_d:
            os.mkdir(f"{in_d}/data")
            copyfile(f"{CURRENT_DIR}/../data/test/train.csv", f"{in_d}/data/train.csv")
            copyfile(f"{CURRENT_DIR}/../data/test/test.csv", f"{in_d}/data/test.csv")
            preprocessing.preprocess(input_dir=in_d, output_dir=out_d)
        assert "train.csv" in os.listdir(f"{out_d}")
        assert "test.csv" in os.listdir(f"{out_d}")
