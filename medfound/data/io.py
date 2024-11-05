from pathlib import Path
import pandas as pd


def read_data(path, **kargs):
    """Read data from different file formats.

    Args:
        path (str): The path to the data file.

    Raises:
        ValueError: If the file format is not supported.

    Returns:
        pandas.DataFrame: The DataFrame containing the data.
    """
    path = Path(path)
    if path.name.endswith('.csv.zip'):
        df_all = pd.read_csv(path, lineterminator='\n', **kargs)
    elif path.name.endswith('.json.zip'):
        df_all = pd.read_json(path, **kargs)
    elif path.name.endswith('.jsonl.zip'):
        df_all = pd.read_json(path, lines=True, **kargs)
    elif path.suffix == '.csv':
        df_all = pd.read_csv(path, lineterminator='\n', **kargs)
    elif path.suffix == '.json':
        df_all = pd.read_json(path, **kargs)
    elif path.suffix == '.jsonl':
        df_all = pd.read_json(path, lines=True, **kargs)
    else:
        raise ValueError(f'Unsupported file format: {path.suffix}')
    return df_all


def export_jsonl(df_data, path):
    """Export DataFrame to JSON Lines format

    Args:
        df_data (pandas.DataFrame): The DataFrame to export.
        path (str): The path to save the JSON Lines file.
    """
    df_data.to_json(path, orient='records', force_ascii=False, lines=True)
