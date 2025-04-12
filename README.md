# Crystall Generation with LLMs

This repository contains the implementation of a project focused on generating crystal structures using Large Language Models (LLMs).

## Project Overview

The main goal of this project is to develop a model capable of generating crystal structures based on textual descriptions using modern machine learning approaches. The project implements the concept of inverse design of materials, creating crystal structures with specified properties from semantic descriptions.

Key achievements:
- Developed a model that can generate valid crystal structures in SLICE format with up to 97.07% validity
- Created a unified dataset genCry containing 315,397 unique crystal structures described by 85 attributes
- Implemented a public web interface for easy access to the model

## Datasets

All information about datasets is available in the `data/` directory. We use open-source datasets, and their citations are provided in the same path.

### Main Dataset: genCry
- Contains 315,397 unique crystal structures
- Described by 85 attributes
- Combines data from multiple sources including Alexandria, MP-20, MPTS-52, JARVIS, and others
- Available at: [Dataset](data/)

You don't need to download them again—just unpack a few of them. Learn more about our datasets by visiting the [Datasets](data/) page.

## Models

We fine-tuned two main models for this project:
1. **Llama 3.2 (3B parameters)**
2. **GPT-2 (1.5B parameters)**

Both models were fine-tuned on our custom dataset using LoRA (Low-Rank Adaptation) technique for efficient training.

More information about models you can find by visiting the [Models](models/) page

## Installation

### Cloning the Repository
First, clone the repository:
```bash
git clone https://github.com/Gruz2520/crystall_generation.git
cd crystall_generation
```

### Installing Dependencies
Install the required dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Unpacking Data and Models
To unpack all datasets and models, you can use the provided unpacking script.

Basic unpacking:
```bash
python notebooks/scr/unpacking.py
```

With dataset filtering:
```bash
python notebooks/scr/unpacking.py -f
```
The `-f` or `--filter` flag enables dataset filtering for the main dataset.

## Web Interface
A public web interface is available for testing and using the model:
[Crystal Generation Web App](https://gencry.streamlit.app/)

## Citation
If you use this work, please cite our project appropriately.

## Additional Information
- The project successfully combines dataset expansion with contextual prompts to achieve optimal results
- Best performance was achieved using SLICE format for output and structured input format with contextual hints
- Future work includes improving data quality, exploring more complex architectures, and expanding to other formats like CIF
