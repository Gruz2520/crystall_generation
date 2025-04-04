import os
import pandas as pd
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments

model_name_new = "gpt2_scratch"

# Параметры модели
vocab_size = 50257       # Размер словаря
max_seq_length = 128     # Максимальная длина последовательности
batch_size = 4           # Размер батча
learning_rate = 5e-4     # Скорость обучения
num_epochs = 3           # Количество эпох

# Создание токенизатора
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Шаг 1: Загрузка CSV файла через pandas
train_df = pd.read_csv('data/train_gpt2.csv')
validation_df = pd.read_csv('data/validation_gpt2.csv')

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

# Создание новой конфигурации GPT-2
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=max_seq_length,
    n_embd=768,
    n_layer=6,
    n_head=12
)

# Создание новой модели GPT-2
model = GPT2LMHeadModel(config)

# Настройка параметров обучения
training_args = TrainingArguments(
    output_dir=f"crystall_generation/finetune/results/{model_name_new}",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_dir=f'crystall_generation/finetune/logs/{model_name_new}',
    logging_steps=10,
    fp16=False,
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
os.makedirs("models/gpt2_from_scratch", exist_ok=True)
model.save_pretrained("models/gpt2_from_scratch")
tokenizer.save_pretrained("models/gpt2_from_scratch")

print("Model saved")

# Тестирование модели
from transformers import pipeline

# Загрузка обученной модели
model = GPT2LMHeadModel.from_pretrained("models/gpt2_from_scratch")
tokenizer = AutoTokenizer.from_pretrained("models/gpt2_from_scratch")

print("Start testing the model")

# Тестовый вход
input_text = "band_gap=0.0 e_form=-0.7 ->"
encoded_input = tokenizer(input_text, return_tensors="pt")

# Генерация текста
output = model.generate(**encoded_input, max_length=50)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)