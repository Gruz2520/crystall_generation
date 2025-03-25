import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard

model_name_new = "fine_tuned_gpt2"

# Load LLaMA tokenizer
model_name = "gpt2"  # Specify the actual version of LLaMA
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Добавляем специальный токен для разделения входных данных
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Шаг 2: Подготовка датасета
def preprocess_function(examples):
    inputs = [f"{row['band_gap']} {row['e_form']} -> {row['crystal_structure']}" for row in examples]
    return tokenizer(inputs, truncation=True, padding='max_length', max_length=128)

# Загрузите ваш датасет (например, из CSV файла)
dataset = load_dataset('csv', data_files={'train': 'crystall_generation/data/train_gpt2.csv', 'validation': 'crystall_generation/data/validation_gpt2.csv'})

# Применяем препроцессинг
tokenized_datasets = dataset.map(preprocess_function, batched=True)


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
    fp16=False,  # Use mixed precision for faster training
    push_to_hub=False,
    report_to="tensorboard",  # Specify that we are using TensorBoard
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

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
input_text = "band_gap=0.0 e_form=-0.7"
output = generator(input_text, max_length=50)
print(output)