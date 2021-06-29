from kfp import components


def train(
    input_dir: components.InputPath(str),
    model_dir: components.OutputPath(str),
    n_epochs: int = 10,
    n_samples: int = None,
) -> None:
    import mlflow
    mlflow.autolog()

    import os

    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from tensorflow import keras

    file = "train.csv"
    df = pd.read_csv(os.path.join(input_dir, file))
    df.pop("id")
    if n_samples:
        df = df.sample(n_samples, random_state=42)
    print(f"Example data from {file}:\n{df.head()}\n{df.info()}")

    BATCH_SIZE = 1024

    METRICS = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]
    target = df.pop("Response")
    print(target.describe())
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    train_dataset = dataset.shuffle(len(df)).batch(BATCH_SIZE)
    print(train_dataset)

    #     train, test = \
    #               np.split(df.sample(frac=1, random_state=42),
    #                        [int(.75*len(df))])

    #     train_labels = train["Response"]
    #     train_data = train.drop(["Response"], axis=1)
    #     print(f"Training labels: {train_labels.describe()}")

    #     test_labels = test["Response"]
    #     test_data = test.drop(["Response"], axis=1)

    # A Really simple model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=METRICS,
    )

    # Run a training job with specified number of epochs
    model.fit(train_dataset, epochs=n_epochs)

    #     model.fit(
    #         train_data.to_numpy(),
    #         train_labels.to_numpy(),
    #         batch_size=BATCH_SIZE,
    #         validation_data=(test_data.to_numpy(), test_labels.to_numpy()),
    #         epochs=n_epochs,
    #     )

    # Save the model to the designated dir
    print("Saving model to", model_dir)
    model.save(model_dir)
