from scr.data_utils import calculate_slices_for_dataset
import pandas as pd

data = pd.read_csv(r"data\alexandria\alexandria_full.csv")

calculate_slices_for_dataset(data, 'cif', "data/alexandria/alexandria_full_slice.csv")