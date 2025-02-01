import pandas as pd
import os
from dotenv import load_dotenv
import pymatgen.ext.matproj
from typing import Dict, List, Optional

load_dotenv()

def query_material_project(
    criteria: Dict,
    properties: List[str],
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Queries the Material Project database and returns the results as a DataFrame.
    Optionally saves the data to a CSV file.

    :param criteria: Dictionary with filtering criteria for the query.
    :param properties: List of properties to retrieve from the Material Project.
    :param output_file: Optional path to save the results as a CSV file.
                       If not provided, the data is not saved.
    :return: DataFrame containing the query results.
    :raises ValueError: If the API_KEY is not found in the .env file.
    """
    # Retrieve the API key from the environment variables
    API_KEY = os.getenv("MATERIAL_PROJECT_API_KEY")
    
    if not API_KEY:
        raise ValueError("API_KEY not found in the .env file. Please add MATERIAL_PROJECT_API_KEY to the .env file.")
    
    # Initialize MPRester with the API key
    mpr = pymatgen.ext.matproj.MPRester(API_KEY)
    
    # Query the Material Project database
    data = mpr.query(criteria=criteria, properties=properties)
    
    # Convert the data into a DataFrame
    df_mp = pd.DataFrame(data)
    
    # Save the data to a CSV file if output_file is provided
    if output_file:
        df_mp.to_csv(output_file, index=False)
        print(f"Data saved to file: {output_file}")
    
    return df_mp