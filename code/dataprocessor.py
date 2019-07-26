import pandas as pd
import os.path
import glob
from sklearn.model_selection import train_test_split


def prepare_data():
    path = r'./data/Anon17csv'

    # Get all the CSV files
    all_files = glob.glob(os.path.join(path, "*.csv"))
    # Create dataframes from each object
    df_from_each_file = (pd.read_csv(f, index_col=0) for f in all_files)

    # Create one dataframe with all
    concatenated_df = pd.concat(df_from_each_file)
    print(concatenated_df.columns)
    print(len(concatenated_df.index))

    # Write the data to one file
    concatenated_df.to_csv("./data/Anon17csv/data.csv", columns=concatenated_df.columns)

    # I2P, I2PApp0BW, I2PApp80BW, I2PUsers : Eepsites, pci
    # JonDonoym: JonDonym
    # Tor, TorApp, TorPT: Tor, Browsing, Torrent, Streaming, Meek
    # Jondonym: 1 length(6335)
    # Tor: 2 length(5283)
    # TorApp: 3 length(252)
    # TorPt: 4  length(302240)

# Check if data file exists which it should
# Do train test split here


def get_data():
    if not os.path.exists("./data/Anon17csv/data.csv"):
        print("Preparing Data")
        prepare_data()

    print("Creating train-test split")
    df = pd.read_csv("./data/Anon17csv/data.csv", header=0, index_col=0)
    train, test = train_test_split(df, test_size=0.33, random_state=42)
    print("Writing train split to csv")
    train.to_csv("./data/Anon17csv/train_data.csv", columns=train.columns)
    print("Writing test split to csv")
    test.to_csv("./data/Anon17csv/test_data.csv", columns=test.columns)
