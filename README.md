# SAME Example: Cross-Selling Products and Services

[![Code Quality](https://github.com/SAME-Project/example-kubeflow-CrossSell/actions/workflows/quality.yml/badge.svg)](https://github.com/SAME-Project/example-kubeflow-CrossSell/actions/workflows/quality.yaml) [![Tests](https://github.com/SAME-Project/example-kubeflow-CrossSell/actions/workflows/test.yml/badge.svg)](https://github.com/SAME-Project/example-kubeflow-CrossSell/actions/workflows/test.yml) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)

**TL;DR;** Reproducible condition monitoring project using Kubeflow and SAME, in a single command

- [Installing / Getting Started](#installing--getting-started)
  * [Pipeline Parameters](#pipeline-parameters)
  * [Pipeline Stages](#pipeline-stages)
- [Developing](#developing)
  * [Testing](#testing)
- [Known Issues](#known-issues)
- [Contributing](#contributing)
- [Credits](#credits)

Cross-selling other products and services to current customers is an important and reliable source of commerce. However, time, effort and customer happiness are all impacted if cross-selling randomly. This project uses a reproducible Kubeflow pipeline managed by SAME to provide a robust model trained on a tabular dataset.

## Installing / Getting Started

Create a working SAME installation by [following instructions found in the wiki](https://github.com/azure-octo/same-cli/wiki/Epic-Sprint-1-Demo), but stop before the "Run a program" section. Then run the following commands:

```bash
git clone https://github.com/SAME-Project/example-kubeflow-CrossSell
cd example-kubeflow-CrossSell
same program create -f same.yaml
same program run -f same.yaml --experiment-name example-kubeflow-CrossSell --run-name default
```

Now browse to your kubeflow installation and you should be able to see an experiment and a run.

### Pipeline Parameters

| Pipeline parameter | Description |
| ------ | ------ |
| epochs | Integer. Number of training epochs. (default: 10) |

### Pipeline Stages

#### 1. Download dataset ([code](./src/download.py))
This component, given the dataset url, downloads all its contents inside an OutputPath Artifact.

#### 2. Preprocessing ([code](./src/preprocessing.py))
This component performs the following operations:

    1. Given an InputPath containing the previously downloaded dataset, extracts all the training files (audio), converts them into numeric arrays.
    2. Performs preprocessing to clean the data.

#### 3. Train ([code](./src/train.py))
This component performs the following operations:

    1. Given an InputPath containing the previously downloaded dataset, extracts all the training files (audio), converting them into numeric arrays.
    2. Uses those arrays, trains a model with the specified parameters.
    3. Save the model in an OutputPath Artifact.

#### 4. Metrics ([code](./src/metrics.py))
This component is passed the mlpipelinemetrics which contains metrics and generates a visualization of them that the kubeflow UI can understand.

##### MLflow

This branch supports MLflow tracking so you can watch long-running training metrics live. To install MLflow on your SAME cluster, install Terraform and ensure that either your `KUBECONFIG` environment variable or your `~/.kube/config` file points to the same cluster you ran `same init` on, then run:

```
git clone https://github.com/combinator-ml/terraform-k8s-mlflow
cd terraform-k8s-mlflow
terraform init
terraform apply
```

Then run:
```
kubectl port-forward service/mlflow -n mlflow 5000:5000
```
And open [http://localhost:5000](http://localhost:5000).

## Developing

When attempting to run or test the code locally you will need to install the reqiured libraries (this requires [poetry](https://python-poetry.org)).

```bash
make install
```

### Testing

This repo is not a library, nor is it meant to run with different permutations of Python or library versions. It is not guaranteed to work with different Python or library versions, but it might. There is limited matrix testing in the github action CI/CD.

```bash
make tests
```

## Known Issues

None.

## Contributing

See [CONTRUBUTING.md](CONTRIBUTING.md).

## Credits

This project was delivered by [Winder Research](https://WinderResearch.com), an ML/RL/MLOps consultancy.

Data originally from: https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction Licensed under GPL 2. Who orignally obtained the data from Analytics Vidhya.
