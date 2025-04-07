import os
import py7zr
from pathlib import Path
import pandas as pd
import argparse
import importlib.util

def filter_structures():
    data_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_utils.py'))
    spec = importlib.util.spec_from_file_location("data_utils", data_utils_path)
    data_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_utils)
    data_utils.filter_structures_from_dataframe(dataframe=pd.read_csv("data/genCry_no_filter.csv"), output_file="data/genCry.csv")

def find_multivolume_files(first_volume_path: str) -> list:
    """
    Finds all parts of a multivolume archive based on the first volume.

    :param first_volume_path: Path to the first volume of the multivolume archive.
    :return: List of file paths for all parts of the archive.
    """
    if not os.path.exists(first_volume_path):
        raise FileNotFoundError(f"The first volume file '{first_volume_path}' does not exist.")

    base_path = os.path.dirname(first_volume_path)
    base_name = os.path.basename(first_volume_path).rsplit('.', 1)[0]  # Remove the numeric part (e.g., '.001')

    # Generate potential filenames for the multivolume archive
    filenames = []
    index = 1
    while True:
        filename = f"{base_name}.{index:03d}"  # Format: example.001, example.002, etc.
        full_path = os.path.join(base_path, filename)
        if not os.path.exists(full_path):
            break
        filenames.append(full_path)
        index += 1

    return filenames


def extract_multivolume_archive(first_volume_path: str, output_dir: str) -> None:
    """
    Extracts a multivolume .7z archive by concatenating all parts into a single file.

    :param first_volume_path: Path to the first volume of the multivolume archive.
    :param output_dir: Directory where the files will be extracted.
    """
    temp_archive_path = None 
    try:
        # Find all parts of the multivolume archive
        filenames = find_multivolume_files(first_volume_path)
        if not filenames:
            raise FileNotFoundError("No parts of the multivolume archive found.")

        # Create a temporary single archive file by concatenating all parts
        temp_archive_path = os.path.join(output_dir, "temp_combined_archive.7z")
        with open(temp_archive_path, 'wb') as outfile:  # Open in binary write mode
            for fname in filenames:
                with open(fname, 'rb') as infile:  # Open in binary read mode
                    outfile.write(infile.read())

        # Extract the combined archive
        extract_7z_archive(temp_archive_path, output_dir)

        print(f"Multivolume archive successfully extracted to {output_dir}.")
    except Exception as e:
        print(f"Error extracting multivolume archive: {e}")
    finally:
        if temp_archive_path and os.path.exists(temp_archive_path):
            try:
                os.unlink(temp_archive_path)
                print(f"Temporary archive '{temp_archive_path}' has been deleted.")
            except Exception as e:
                print(f"Failed to delete temporary archive: {e}")

def extract_7z_archive(archive_path: str, output_dir: str) -> None:
    """
    Extracts a .7z archive to the specified directory.

    :param archive_path: Path to the .7z archive.
    :param output_dir: Directory where the files will be extracted.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
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
    
    print("Start extracting model")
    extract_multivolume_archive("models/model_arhive/model.7z.001", "models/fine_tuned_gpt2_on_alex_full/")
    print("All models successfully extracted.")

    if args.filter:
        print("Starting filtering main dataset...")
        filter_structures()
        print("Main dataset successfully filtered. Filtered dataset: data/genCry_f.csv")
    else:
        extract_7z_archive("data/genCry_f.7z", "data/")
        print("Filtering skipped. Use -f to enable filtering.")