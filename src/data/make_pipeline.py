import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('pipeline_path', type=click.Path())
def main(input_path, pipeline_path):
    """ Creates a data processing pipeline that can be used for training and
        validation data.
    """
    # Read the data from CSV file
    df = pd.read_csv(input_path)

    # Drop any rows with missing values
    df = df.dropna()
    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

    # Split the data into training and validation sets
    df_train, _ = train_test_split(df, test_size=0.2, random_state=1)

    # Define the preprocessor for categorical variables
    cat_preds = ["job", "marital", "education", "default", "housing",
                 "loan", "contact", "day", "month", "poutcome"]
    num_preds = ["age", "balance", "duration", "campaign", "pdays",
                 "previous"]
    assert len(cat_preds) + len(num_preds) == 16, "Must have 16 predictors"
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_preds),
            ("cat",
             OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
             cat_preds)
        ],
        remainder="passthrough"
    )

    # Create the pipeline combining preprocessing and model training
    pipeline = Pipeline([("preprocessor", preprocessor)])
    pipeline.fit(df_train)

    dump(pipeline, pipeline_path)


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
