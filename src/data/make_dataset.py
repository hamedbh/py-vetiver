import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('pipeline_path', type=click.Path(exists=True))
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path())
@click.argument('partition', type=str)
def main(pipeline_path, input_data_path, data_path, partition):
    """
    Writes out dataset from input data, processed with the sklearn pipeline.
    """
    pipeline = load(pipeline_path)
    df = pd.read_csv(input_data_path)
    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

    df_train, df_val = train_test_split(df, test_size=0.2, random_state=1)

    assert partition in ['train',
                         'val'], "`partition` must equal 'train' or 'val'"
    if (partition == "train"):
        d = df_train
    elif (partition == "val"):
        d = df_val

    d_trans = pipeline.transform(d).astype(np.float32)
    np.save(data_path, d_trans, allow_pickle=True)


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
