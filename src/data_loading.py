import pandas as pd

files = {
    'casualty': '/content/drive/MyDrive/MINOR_PROJECT_FILES/casualty.csv',
    'collision': '/content/drive/MyDrive/MINOR_PROJECT_FILES/collision.csv',
    'vehicle': '/content/drive/MyDrive/MINOR_PROJECT_FILES/vehicle.csv',
    'final_df': '/content/drive/MyDrive/MINOR_PROJECT_FILES/final_df.parquet',
    'vehicle_agg': '/content/drive/MyDrive/MINOR_PROJECT_FILES/veh_agg.parquet',
    'casualty_agg': '/content/drive/MyDrive/MINOR_PROJECT_FILES/cas_agg.parquet'
}


def data_loading(path: str) -> pd.DataFrame | None:
    """
    Load a CSV or Parquet file into a DataFrame.

    Args:
        path: Full path of a csv or parquet file.

    Returns:
        DataFrame if loaded successfully, otherwise None.

    Raises:
        ValueError: If the file path is empty or the file format is unknown.
    """
    if not path:
        raise ValueError("File path is empty")

    try:
        if path.lower().endswith('.csv'):
            return pd.read_csv(path, low_memory=False)
        elif path.lower().endswith('.parquet'):
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unknown format: {path}")

    except FileNotFoundError:
        print(f"File not found: {path}")
    except pd.errors.EmptyDataError:
        print(f"File is empty: {path}")
    except ValueError:
        raise
    except Exception as e:
        print(f"Error loading {path}: {e}")

    return None


if __name__ == '__main__':
    casualty_df = data_loading(files['casualty'])
    collision_df = data_loading(files['collision'])
    vehicle_df = data_loading(files['vehicle'])
    vehicle_agg_df = data_loading(files['vehicle_agg'])
    casualty_agg_df = data_loading(files['casualty_agg'])
    final_df = data_loading(files['final_df'])
