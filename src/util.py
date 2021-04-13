from kfp import components


def generate_submission(
    input_dir: components.InputPath(str),
    model_dir: components.InputPath(str),
    result_dir: components.OutputPath(str),
) -> None:
    import os

    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from tensorflow.python import keras

    file = "test.csv"
    df = pd.read_csv(os.path.join(input_dir, file))
    ids = df["id"]
    df = df.drop(["id", "Response"], axis=1)
    print(f"Example data from {file}:\n{df.head()}\n{df.info()}")

    # Load the model
    model = keras.models.load_model(model_dir)

    # Display the model's architecture
    model.summary()

    # Generate the predictions
    predictions = tf.round(model.predict(df.to_numpy(), verbose=2))
    print(f"Prediction shape: {predictions.shape}")

    # Convert to series for csv creation
    predictions = pd.Series(predictions[:, 0])
    print(f"Prediction shape: {predictions.describe()}")

    # Format the output file
    submission = pd.concat(
        [ids, predictions], axis=1, ignore_index=True, keys=["id", "Response"]
    )
    print(submission.head())
    submission.to_csv(result_dir, index=False)
