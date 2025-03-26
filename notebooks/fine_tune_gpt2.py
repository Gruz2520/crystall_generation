import os
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name_new = "fine_tuned_gpt2_on_alex_full"

# Load GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Добавляем специальный токен для разделения входных данных
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Шаг 1: Загрузка CSV файла через pandas
train_df = pd.read_csv('crystall_generation/data/train_alex_full_gpt2.csv')
validation_df = pd.read_csv('crystall_generation/data/validation_alex_full_gpt2.csv')

# Преобразование DataFrame в Dataset
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)

# Шаг 2: Подготовка датасета
def preprocess_function(examples):
    # Формирование входных данных
    inputs = [f"{inp} ->" for inp in examples['input']]
    targets = examples['output']
    
    # Токенизация входных данных
    model_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=128)
    
    # Токенизация целевых меток
    labels = tokenizer(targets, truncation=True, padding='max_length', max_length=128)
    
    # Добавление labels в модельные данные
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs
# Применяем препроцессинг
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True)

# Создание директорий для логов и результатов
os.makedirs(f"crystall_generation/finetune/results/{model_name_new}", exist_ok=True)
os.makedirs(f"crystall_generation/finetune/logs/{model_name_new}", exist_ok=True)

# Настройка параметров обучения
training_args = TrainingArguments(
    output_dir=f"crystall_generation/finetune/results/{model_name_new}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=20,
    logging_dir=f'crystall_generation/finetune/logs/{model_name_new}',
    logging_steps=100,
    fp16=True,
    push_to_hub=False,
    report_to="tensorboard",
)

# Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
)

print("Start model training")

# Начало обучения
trainer.train()

print("Model successfully trained")

# Сохранение модели и токенизатора
model.save_pretrained(f"models/{model_name_new}")
tokenizer.save_pretrained(f"models/{model_name_new}")

print("Model saved")

# Тестирование модели
from transformers import pipeline

# Загрузка дообученной модели
model = AutoModelForCausalLM.from_pretrained(f"models/{model_name_new}")
tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name_new}")

print("Start testing the model")

# Тестовый вход
input_text = "0.0 -0.7 ->"
encoded_input = tokenizer(input_text, return_tensors="pt")

# Генерация текста
output = model.generate(**encoded_input, max_length=50)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)