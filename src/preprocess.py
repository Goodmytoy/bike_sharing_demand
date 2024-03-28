import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

#drop_features = 'datetime'

def datetime_handler(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """`datetime 처리 함수

    Args:
        df (pd.DataFrame): 데이터프레임

    Returns:
        pd.DataFrame: datetime 처리 후 데이터
    """
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour

    df = df.drop(col, axis=1)

    return df

preprocess_pipeline = ColumnTransformer(
    transformers=[
        (
            "datetime_handler",
            FunctionTransformer(datetime_handler, kw_args={"col": "datetime"}),
            ['datetime'],
        )],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(transform="pandas")
