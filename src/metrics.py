from typing import NamedTuple

from kfp import components


def produce_metrics(
    input_dir: components.InputPath(str), model_dir: components.InputPath(str)
) -> NamedTuple("Outputs", [("mlpipeline_metrics", "Metrics")]):  # noqa: F821
    import json
    import os

    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from tensorflow import keras

    BATCH_SIZE = 1024

    file = "train.csv"
    df = pd.read_csv(os.path.join(input_dir, file))
    df.pop("id")
    target = df.pop("Response")
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values)).batch(
        BATCH_SIZE
    )

    # Load the model
    model = keras.models.load_model(model_dir)

    # Evaluate the model and print the results
    print("__Evaluation results__")
    evaluation_results = model.evaluate(dataset, verbose=2)
    metrics_list = []
    for name, value in zip(model.metrics_names, evaluation_results):
        print(name, ": ", value)
        metrics_list.append(
            {
                "name": name,  # The name of the metric. Visualized as the column name in the runs table.
                "numberValue": value,  # The value of the metric. Must be a numeric value.
                "format": "PERCENTAGE",  # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
            }
        )

    return [json.dumps({"metrics": metrics_list})]
