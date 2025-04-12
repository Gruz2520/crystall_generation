# Dataset genCry
Our dataset is a combination of datasets such as: [Alexandria](alexandria/), [carbon_24](carbon_24/), [Jarvis](Jarvis/), [MP_20](mp_20/), [MPTS_52](mpts_52/), [perov_5](perov_5/). The page for each dataset gives you a better look at it. These datasets were chosen because they contain relaxed stable crystals. The stability of the crystals will increase the accuracy of generating stable compounds later on. There are 315,397 crystals in the dataset, with 85 attributes. For training we used the format that you can find below.

| input| output |
|:-----:|:-----:|
| Physicochemical characteristics of structure | Text representation of the structure |

Filtering by original dataset was performed to exclude duplicates, crystals with less than 2 atoms, and broken CIF files. The definition of a broken CIF file was done by converting to Structure from the [pymatgen](https://pymatgen.org/) library. The filtering algorithm can be found [here.](../notebooks/scr/data_utils.py)

In rep you will find genCry.7z and genCry_f.7z You can extend them yourself or use or [unpacking script](#unpacking). genCry_f.7z contains the original dataset, the other file contains the merged dataset without filters.

train & valid datasets you also can find in this dir.

## Visualization

<p align="center">
  <img src="../plots/proportion_of_datasets.png" />
</p>

## Unpacking

To unpack all datasets, you can use the unpacking script.  
If you want to simply unpack the datasets, run the script without any parameters:
```bash
python notebooks/scr/unpacking.py
```
Add the `-f` or `--filter` flag to enable dataset filtering for the main dataset:
```bash
python notebooks/scr/unpacking.py -f
```