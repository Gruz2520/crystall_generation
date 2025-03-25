import os
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

model_name_new = "fine_tuned_gpt2"

# Load GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Добавляем специальный токен для разделения входных данных
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Шаг 2: Подготовка датасета
def preprocess_function(examples):
    inputs = [f"{inp} -> {out}" for inp, out in zip(examples['input'], examples['output'])]
    return tokenizer(inputs, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

# Загрузите ваш датасет (например, из CSV файла)
dataset = load_dataset('csv', data_files={
    'train': 'crystall_generation/data/train_gpt2.csv',
    'validation': 'crystall_generation/data/validation_gpt2.csv'
})

# Применяем препроцессинг
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Create directories for logs and results
os.makedirs(f"crystall_generation/finetune/results/{model_name_new}", exist_ok=True)
os.makedirs(f"crystall_generation/finetune/logs/{model_name_new}", exist_ok=True)

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
    logging_dir=f'crystall_generation/finetune/logs/{model_name_new}',
    logging_steps=10,
    fp16=False,
    push_to_hub=False,
    report_to="tensorboard",
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
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

# Test input
input_text = "0.0 -0.7 ->"
encoded_input = tokenizer(input_text, return_tensors="pt")

# Generate text
output = model.generate(**encoded_input, max_length=50)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)