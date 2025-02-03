import os
import py7zr
from pathlib import Path
from data_utils import filter_structures_from_dataframe
import pandas as pd
import argparse

def extract_7z_archive(archive_path: str, output_dir: str) -> None:
    """
    Extracts a .7z archive to the specified directory.

    :param archive_path: Path to the .7z archive.
    :param output_dir: Directory where the files will be extracted.
    """
    try:
        # Open the .7z archive and extract its contents to the output directory
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=output_dir)
        print(f"Archive {archive_path} successfully extracted to {output_dir}.")
    except Exception as e:
        print(f"Error extracting archive {archive_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and filter datasets.")
    # Add a flag '-f' or '--filter' to enable dataset filtering
    parser.add_argument('-f', '--filter', action='store_true', help="Enable filtering of the main dataset.")
    args = parser.parse_args()

    extract_7z_archive("data/alexandria/alexandria_pbe.7z", "data/alexandria/")
    extract_7z_archive("data/Jarvis/jarvis.7z", "data/Jarvis/")
    extract_7z_archive("data/genCry.7z", "data/")

    print("All data successfully extracted.")

    if args.filter:
        print("Starting filtering main dataset...")
        filter_structures_from_dataframe(dataframe=pd.read_csv("data/genCry_no_filter.csv"), output_file="data/genCry.csv")
        print("Main dataset successfully filtered. Filtered dataset: data/genCry_f.csv")
    else:
        extract_7z_archive("data/genCry_f.7z", "data/")
        print("Filtering skipped. Use -f to enable filtering.")