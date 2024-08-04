import pandas as pd
import os

def load_data():
    """
    Loads the datasets from the provided URLs.
    
    :return: Tuple of DataFrames (pred_universe_raw, arrest_events_raw)
    """
    pred_universe_url = 'https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1'
    arrest_events_url = 'https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1'
    
    # Try reading as CSV due to the format issue
    pred_universe_raw = pd.read_csv(pred_universe_url)
    arrest_events_raw = pd.read_csv(arrest_events_url)
    
    return pred_universe_raw, arrest_events_raw

def preprocess_data(pred_universe_raw, arrest_events_raw):
    """
    Preprocess the previously loaded datasets. Converts the date columns and drops the columns that aren't needed.
    
    :param pred_universe_raw: DataFrame of the pred_universe dataset
    :param arrest_events_raw: DataFrame of the arrest_events dataset
    :return: Tuple of preprocessed DataFrames (pred_universe, arrest_events)
    """
    # Convert date columns to datetime format
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['filing_date'])
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['filing_date'])
    
    # Drop the original filing_date columns
    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)
    
    return pred_universe_raw, arrest_events_raw

def save_data(pred_universe, arrest_events, data_dir='./data'):
    """
    Saves the datasets to CSV format.
    
    :param pred_universe: Preprocessed DataFrame of the pred_universe dataset
    :param arrest_events: Preprocessed DataFrame of the arrest_events dataset
    :param data_dir: Directory to save the CSV files
    """
    os.makedirs(data_dir, exist_ok=True)
    
    pred_universe_path = os.path.join(data_dir, 'pred_universe_raw.csv')
    arrest_events_path = os.path.join(data_dir, 'arrest_events_raw.csv')
    
    pred_universe.to_csv(pred_universe_path, index=False)
    arrest_events.to_csv(arrest_events_path, index=False)
    
    print(f"Data saved to {data_dir}")

def main():
    """
    Main function to perform ETL and save the processed data.
    """
    # Load the raw data
    pred_universe_raw, arrest_events_raw = load_data()
    
    # Preprocess the data
    pred_universe, arrest_events = preprocess_data(pred_universe_raw, arrest_events_raw)
    
    # Save the preprocessed data
    save_data(pred_universe, arrest_events)

if __name__ == "__main__":
    main()