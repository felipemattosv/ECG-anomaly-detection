import pandas as pd
import re
from sklearn.preprocessing import StandardScaler

def load_data(train_data_path: str, val_data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation parquet files, select time series columns (t1..t140),
    and flatten each into a single 'signal' column.
    """
    train_df = pd.read_parquet(train_data_path)
    val_df = pd.read_parquet(val_data_path)

    input_columns = [col for col in train_df.columns if re.match(r't\d+$', col)]
    train_df = train_df[input_columns]
    val_df = val_df[input_columns]

    train_df = pd.DataFrame(train_df.values.flatten(), columns=['signal'])
    val_df = pd.DataFrame(val_df.values.flatten(), columns=['signal'])

    return train_df, val_df

def scale_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on train data, transform both train and val.
    Returns scaled dataframes and the fitted scaler (needed for model saving).
    """
    scaler = StandardScaler()
    train_df_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    val_df_scaled = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns, index=val_df.index)
    return train_df_scaled, val_df_scaled, scaler
