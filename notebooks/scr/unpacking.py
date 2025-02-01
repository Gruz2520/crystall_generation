import os
import py7zr
from pathlib import Path
from data_utils import filter_structures_from_dataframe
import pandas as pd

def extract_7z_archive(archive_path: str, output_dir: str) -> None:
    """
    Extracts a .7z archive to the specified directory.

    :param archive_path: Path to the .7z archive.
    :param output_dir: Directory where the files will be extracted.
    """
    try:
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=output_dir)
        print(f"Archive {archive_path} successfully extracted to {output_dir}.")
    except Exception as e:
        print(f"Error extracting archive {archive_path}: {e}")
        

if __name__ == '__main__':
    # unpacking all datasets
    extract_7z_archive("data/alexandria/alexandria_pbe.7z", "data/alexandria/")
    extract_7z_archive("data/Jarvis/jarvis.7z", "data/Jarvis/")
    extract_7z_archive("data/genCry.7z", "data/")
    
    print("All data successfully extracted.")
    print("starting filtering main dataset...")
    
    # filtering dataset
    filter_structures_from_dataframe(dataframe = pd.read_csv("data/genCry.csv"), output_file = "data/genCry_f.csv")
    print("Main dataset successfully filtered. Filtered dataset: data/genCry_f.csv")