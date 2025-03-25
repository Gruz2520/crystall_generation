import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard

# Load CSV file
df = pd.read_csv('crystall_generation/data/df_for_llama_bg_eform_slice.csv')

model_name_new = "fine_tuned_Llama-3.2-3b_bg_eform"

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Load LLaMA tokenizer
model_name = "crystall_generation/models/llama"  # Specify the actual version of LLaMA
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a special EOS (End of Sentence) token
tokenizer.pad_token = tokenizer.eos_token

# Function for tokenizing data
def preprocess_function(examples):
    inputs = examples['input']
    outputs = examples['output']
    
    # Tokenize input and output
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(outputs, max_length=512, truncation=True, padding='max_length')
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Apply preprocessing function to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# If you want to use LoRA (Low-Rank Adaptation) to save resources:
lora_config = LoraConfig(
    r=8,  # Rank of the matrix
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Target layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=f"crystall_generation/finetune/results/{model_name_new}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_dir=f'crystall_generation/finetune/logs/{model_name_new}',  # Directory for TensorBoard logs
    logging_steps=10,  # Frequency of logging
    fp16=True,  # Use mixed precision for faster training
    push_to_hub=False,
    report_to="tensorboard",  # Specify that we are using TensorBoard
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("Start model training")

# Start training
trainer.train()

print("Model successfully trained")

# Save the model and tokenizer
model.save_pretrained(f"models/{model_name_new}")
tokenizer.save_pretrained(f"models/{model_name_new}")

print("Model saved")

# Test the model
from transformers import pipeline

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained(f"models/{model_name_new}")
tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name_new}")

print("Start testing the model")

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
input_text = "band_gap=0.0 e_form=-0.7"
output = generator(input_text, max_length=50)
print(output)