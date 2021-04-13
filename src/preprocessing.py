from kfp import components


def preprocess(
    input_dir: components.InputPath(str),
    output_dir: components.OutputPath(str),
    read_csv_opts: dict = {},
) -> None:
    import os

    import pandas as pd

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
        df["Annual_Premium"] = (
            df["Annual_Premium"] / 1000000
        )  # Gross, but was affecting performance
        return df

    file = "train.csv"
    df = load_and_process_file(file)

    # Balance training data
    ### Separate the majority and minority classes
    df_minority = df[df["Response"] == 1]
    df_majority = df[df["Response"] == 0]

    # Downsample majority class to the number of observations in the minority class
    df_majority = df_majority.sample(len(df_minority), random_state=42)

    # concat the majority and minority dataframes
    df = pd.concat([df_majority, df_minority])

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
    _, df = train.align(df, join="left", axis=1, fill_value=0)

    print(f"Example data from {file}:\n{df.head()}\n{df.info()}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, file)
    print(f"Saving data to {output_csv}")
    df.to_csv(output_csv, index=False)
