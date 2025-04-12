# Models

This directory contains the fine-tuned models developed for crystal structure generation. The most successful model, `gpt2_with_alex_p`, is described in detail below along with its training parameters and performance metrics.

## Model Overview: gpt2_with_alex_p

The `gpt2_with_alex_p` model represents our best-performing implementation of crystal structure generation from textual descriptions. This model was fine-tuned using:

- Extended dataset (`genCry + Alexandria`)
- Contextual prompts added to input data
- Optimized hyperparameters

### Key Features

| Feature | Description |
|---------|-------------|
| Base Architecture | GPT-2 (1.5B parameters) |
| Training Dataset | genCry + Alexandria |
| Input Format | Structured prompt + physical parameters |
| Output Format | SLICE |
| Best Validation Accuracy | 97.07% |

## Training Hyperparameters

The model was fine-tuned using the following configuration:

### Core Training Parameters

| Parameter | Value |
|-----------|-------|
| Evaluation Strategy | epoch |
| Learning Rate | 2e-5 |
| Training Batch Size | 24 |
| Evaluation Batch Size | 24 |
| Number of Epochs | 3 |
| Mixed Precision (fp16) | True |

## Performance Metrics

### Validation Accuracy Comparison

| Model Variant | Average Validity (%) |
|---------------|----------------------|
| gpt2 (baseline) | 93.51% |
| gpt2_with_alex | 95.07% |
| gpt2_p | 96.37% |
| gpt2_with_alex_p | 97.07% |

### Sample Efficiency

| Samples Generated | gpt2_with_alex_p Validity (%) | Baseline Validity (%) |
|--------------------|-------------------------------|------------------------|
| 1 | 97.07% | 93.51% |
| 5 | 99.2% | 96.8% |
| 9 | 100% | 99.1% |

## Graphical Analysis

### Training Loss Curve

![Training Loss Curve](../plots\train_loss_of_model.png)

_Description: The training loss stabilized at approximately 1.77 after several thousand steps, indicating successful adaptation to the extended dataset._

### Sample Efficiency Graph

![Sample Efficiency](../plots\validation_comparison_plot.png)

_Description: The graph demonstrates how validity improves with increased sample generation. The `gpt2_with_alex_p` model reaches 100% validity with 9 samples._

## Usage Instructions

### Loading the Model

You can load model after unpack it with [unpacking script](../README.md#unpacking-data-and-models)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/fine_tuned_gpt2_on_alex_full")
tokenizer = AutoTokenizer.from_pretrained("models/fine_tuned_gpt2_on_alex_full")
```

### Generating Crystal Structures

```python
input_text = "0.3-0.7->"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Using Contextual Prompts

```python
prompt = "Generate a crystal with high stability and reflection index "
input_text = prompt + "0.3 -0.7->"
```

## Recommendations

- Generate multiple samples per request and select the most valid result
- Use temperature adjustment for more diverse results
- Consider post-processing generated structures for additional validation

