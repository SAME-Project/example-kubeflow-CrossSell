# SAME Example: Cross-Sell Prediction

> **This is a work in progress!**

## Usage

Create a working SAME installation by [following instructions found in the wiki](https://github.com/azure-octo/same-cli/wiki/Epic-Sprint-1-Demo), but stop before the "Run a program" section. Then run the following commands:

```bash
git clone https://github.com/SAME-Project/example-kubeflow-CrossSell
cd example-kubeflow-CrossSell
same program create -f same.yaml
same program run -f same.yaml --experiment-name cross-sell --run-name default
```

Now browse to your kubeflow installation and you should be able to see an experiment and a run.