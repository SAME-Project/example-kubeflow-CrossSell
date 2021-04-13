from kfp import components


def extract_tar_from_url(data_dir: components.OutputPath(str)):
    """Download data to the KFP volume to share it among all steps"""
    import logging
    import os
    import tarfile
    import urllib.request

    logger = logging.getLogger(__name__)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = "https://raw.githubusercontent.com/SAME-Project/example-kubeflow-CrossSell/main/data/external/data.tar.gz"
    logger.info(f"Downloading {url}")
    stream = urllib.request.urlopen(url)
    tar = tarfile.open(fileobj=stream, mode="r|gz")
    logger.info(f"Extracting to {data_dir}")
    tar.extractall(path=data_dir)
