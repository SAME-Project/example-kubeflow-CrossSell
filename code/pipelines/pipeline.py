#!/usr/bin/env python
# coding: utf-8
from typing import NamedTuple
from datetime import datetime
import kfp
import kfp.components as components
import kfp.dsl as dsl


# In[236]:


def download(data_dir: components.OutputPath(str)):
    """Download data to the KFP volume to share it among all steps"""
    import urllib.request
    import tarfile
    import os

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = "https://raw.githubusercontent.com/SAME-Project/example-kubeflow-CrossSell/main/code/data/external/data.tar.gz"
    print(f"Downloading {url}")
    stream = urllib.request.urlopen(url)
    tar = tarfile.open(fileobj=stream, mode="r|gz")
    print(f"Extracting to {data_dir}")
    tar.extractall(path=data_dir)


# In[237]:


def preprocess(input_dir: components.InputPath(str), output_dir: components.OutputPath(str), read_csv_opts: dict = {}) -> None:
    import pandas as pd
    import os
    print(f"input_dir: {input_dir}. Contents: {os.listdir(os.path.join(input_dir))}")
    
    def load_and_process_file(file):
        csv_file = os.path.join(input_dir, "data", file)
        print(f"Loading rows from {csv_file} with options {read_csv_opts}")
        df = pd.read_csv(csv_file, **read_csv_opts)
        df = df.drop(["Policy_Sales_Channel"], axis=1)
        for field in ["Gender", "Vehicle_Age", "Vehicle_Damage", "Region_Code"]:
            new_dummies = pd.get_dummies(df[field], prefix=field)
            print(f"{len(new_dummies.columns)} columns encoded from {field}")
            df = df.join(new_dummies)
            df = df.drop([field], axis=1)
        df["Annual_Premium"] = df["Annual_Premium"] / 1000000 # Gross, but was affecting performance
        return df

    file = "train.csv"
    df = load_and_process_file(file)
    
    # Balance training data
    ### Separate the majority and minority classes
    df_minority  = df[df['Response']==1]
    df_majority = df[df['Response']==0]

    # Downsample majority class to the number of observations in the minority class
    df_majority = df_majority.sample(len(df_minority), random_state=42)

    # concat the majority and minority dataframes
    df = pd.concat([df_majority,df_minority])

    # Shuffle data
    df = df.sample(frac=1, random_state=42)
    print(f"Example data from {file}:\n{df.head()}\n{df.info()}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, file)
    print(f"Saving data to {output_csv}")
    df.to_csv(output_csv, index=False)
    
    train = df
    
    file = "test.csv"
    df = load_and_process_file(file)
    _, df = train.align(df, join='left', axis=1, fill_value=0)
        
    print(f"Example data from {file}:\n{df.head()}\n{df.info()}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, file)
    print(f"Saving data to {output_csv}")
    df.to_csv(output_csv, index=False)
        


# In[280]:


def train(input_dir: components.InputPath(str), model_dir: components.OutputPath(str), n_epochs: int = 10, n_samples: int = None) -> None:
    import os
    import tensorflow as tf
    from tensorflow import keras
    import pandas as pd
    import numpy as np

    file = "train.csv"
    df = pd.read_csv(os.path.join(input_dir, file))
    ids = df.pop('id')
    if n_samples:
        df = df.sample(n_samples, random_state=42)
    print(f"Example data from {file}:\n{df.head()}\n{df.info()}")

    BATCH_SIZE = 1024

    METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]
    target = df.pop('Response')
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
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
      ])
    model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=METRICS)

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
    model.save(model_dir)


# In[281]:


def produce_metrics(input_dir: components.InputPath(str), model_dir: components.InputPath(str)) -> NamedTuple('Outputs', [
  ('mlpipeline_metrics', 'Metrics'),
]):
    import json
    import os
    import tensorflow as tf
    from tensorflow import keras
    import pandas as pd
    import numpy as np
    
    BATCH_SIZE = 1024

    file = "train.csv"
    df = pd.read_csv(os.path.join(input_dir, file))
    ids = df.pop('id')
    target = df.pop('Response')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values)).batch(BATCH_SIZE)

    # Load the model
    model = keras.models.load_model(model_dir)

    # Evaluate the model and print the results
    print("__Evaluation results__")
    evaluation_results = model.evaluate(dataset, verbose=2)
    metrics_list = []
    for name, value in zip(model.metrics_names, evaluation_results):
        print(name, ': ', value)
        metrics_list.append({
          'name': name, # The name of the metric. Visualized as the column name in the runs table.
          'numberValue':  value, # The value of the metric. Must be a numeric value.
          'format': "PERCENTAGE",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
        })
    
    return [json.dumps({
    'metrics': metrics_list
    })]


# In[282]:


def generate_submission(input_dir: components.InputPath(str), model_dir: components.InputPath(str), result_dir: components.OutputPath(str)) -> None:
    import os
    import tensorflow as tf
    from tensorflow.python import keras
    import pandas as pd
    import numpy as np

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
    predictions = pd.Series(predictions[:,0])
    print(f"Prediction shape: {predictions.describe()}")
    
    # Format the output file
    submission = pd.concat([ids, predictions], axis=1, ignore_index=True, keys=["id", "Response"])
    print(submission.head())
    submission.to_csv(result_dir, index=False)


# In[287]:


@dsl.pipeline(
    name="Pipeline",
    description="",
)
def pipeline(
    epochs: int = 10,
):  
    BASE_IMAGE = "gcr.io/deeplearning-platform-release/tf2-cpu.2-4:m65"

    downloadOp = components.func_to_container_op(
        download, base_image=BASE_IMAGE
    )()

    preprocessOp = components.func_to_container_op(preprocess, base_image=BASE_IMAGE)(
       downloadOp.output
    )
    
    trainOp = components.func_to_container_op(train, base_image=BASE_IMAGE)(
       preprocessOp.output, n_epochs=epochs,
    )
    
    metricsOp = components.func_to_container_op(produce_metrics, base_image=BASE_IMAGE)(
       preprocessOp.output, trainOp.output
    )
    
    submissionOp = components.func_to_container_op(generate_submission, base_image=BASE_IMAGE)(
       preprocessOp.output, trainOp.output
    )
