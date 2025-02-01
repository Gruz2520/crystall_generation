import pandas as pd
from typing import List, Optional

def combine_csv_files(
    file_paths: List[str], 
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Combines multiple CSV files into a single DataFrame and optionally saves the result to a new CSV file.

    :param file_paths: List of paths to the CSV files to be combined.
    :param output_file: Path to the file where the combined DataFrame will be saved. If None, the file is not saved.
    :return: Combined DataFrame.
    """
    # Read all CSV files into a list of DataFrames
    dfs = [pd.read_csv(file) for file in file_paths]
    
    # Combine all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save the result to a CSV file if output_file is provided
    if output_file:
        combined_df.to_csv(output_file, index=False)
        print(f"Combined data saved to file: {output_file}")
    
    return combined_df