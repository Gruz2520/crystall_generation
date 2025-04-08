# Crystall Generation with LLMs
This repo for project about generation crystall structures with LLM.

### Datasets
All information about datasets is directly available in `data/`. We use open-source datasets, and their citations are available in the same path. You don't need to download them again—just unpack a few of them. You can unpack them using our [unpack script](#unpacking).

Learn more about our dataset **genCry** and others by visiting the [Datasets](data/) page.

### Models

In our project we had used a llama3.2 3b and gpt2 1.5b models for fine-tuned on our dataset

## Installation

### Cloning
First do a cloning of the repository.
```bash
git clone https://github.com/Gruz2520/crystall_generation.git
```
Change directory of the cloned repository:
```bash
cd crystall_generation
```
Install the required dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Unpacking
To unpack all datasets and models, you can use the unpacking script.  
If you want to simply unpack the datasets with model, run the script without any parameters:
```bash
python notebooks/scr/unpacking.py
```
Add the `-f` or `--filter` flag to enable dataset filtering for the main dataset:
```bash
python notebooks/scr/unpacking.py -f
```

## Citation