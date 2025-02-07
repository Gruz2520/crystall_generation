import pandas as pd
from typing import List, Optional
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from io import StringIO
from tqdm import tqdm
import warnings
from scr.invcryrep.invcryrep import InvCryRep

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


def filter_structures_from_dataframe(
    dataframe: pd.DataFrame,
    cif_column: str = "cif",
    material_id_column: str = "material_id",
    ref_column: str = "ref",
    num_of_atoms: int = 2,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Filters structures from a DataFrame containing CIF strings based on the number of atoms.
    Returns a new DataFrame with only the structures that have more than the specified number of atoms.

    :param dataframe: Input DataFrame containing CIF strings and material IDs.
    :param cif_column: Name of the column in the DataFrame that contains the CIF strings. Default is "cif".
    :param material_id_column: Name of the column in the DataFrame that contains the material IDs. Default is "material_id".
    :param num_of_atoms: Minimum number of atoms a structure must have to be included in the result. Default is 2.
    :param ref_column: Name of the column in the DataFrame that contains reference data. Default is "ref".
    :param output_file: Path to the file where the filtered DataFrame will be saved. If None, the file is not saved.
    :return: A DataFrame containing the filtered structures with their material IDs and CIF strings.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    
    filtered_data = []

    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        mat_id = row[material_id_column]
        cif_string = row[cif_column]
        ref = row[ref_column]

        try:
            # Convert the CIF string into a file-like object for parsing
            cif_file_like = StringIO(cif_string)

            # Parse the CIF string into a pymatgen Structure object
            parser = CifParser(cif_file_like)
            structure = parser.get_structures()[0]

            # Check if the structure has more than the specified number of atoms
            if len(structure) > num_of_atoms:
                filtered_data.append({
                    material_id_column: mat_id,
                    cif_column: cif_string,
                    ref_column: ref
                })
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    filtered_df = pd.DataFrame(filtered_data).reset_index().drop("index", axis=1)
    
    if output_file:
        filtered_df.to_csv(output_file, index=False)
        print(f"Filtered data saved to file: {output_file}")
    
    return filtered_df


def calculate_slices_for_dataset(
    dataset: pd.DataFrame,
    cif_column: str = "cif",
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculates SLICES for each CIF file in the dataset and adds the results to the DataFrame.
    Optionally saves the resulting DataFrame to a specified path.

    :param dataset: Input DataFrame containing CIF strings.
    :param cif_column: Name of the column in the DataFrame that contains the CIF strings. Default is "cif".
    :param output_file: Path to save the resulting DataFrame as a CSV file. If None, the file is not saved.
    :return: A DataFrame with added columns for SLICES (SLICE, SLICE PLUS).
    """
    backend = InvCryRep()

    # Add new columns for SLICES
    dataset['SLICE'] = None
    dataset['SLICE PLUS'] = None

    for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Converting CIF to SLICE"):
        cif = row[cif_column]
        try:
            # Calculate SLICES for the current CIF file
            slice_part1, _, final_slice = backend.concatenate_slices(cif)
            
            dataset.at[index, 'SLICE'] = slice_part1
            dataset.at[index, 'SLICE PLUS'] = final_slice
        except Exception as e:
            print(f"Error processing CIF file at row {index}: {e}")
            dataset.at[index, 'SLICE'] = None
            dataset.at[index, 'SLICE PLUS'] = None

    # Save the result to a file if a path is provided
    if output_file:
        dataset.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    return dataset