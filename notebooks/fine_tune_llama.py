import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer

# Загрузка CSV файла
df = pd.read_csv('crystall_generation/data/df_for_llama_m100.csv')

# Преобразование DataFrame в Dataset
dataset = Dataset.from_pandas(df)

# Загрузка токенизатора LLaMA
model_name = "crystall_generation/models/llama"  # Укажите актуальную версию LLaMA
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Добавление специального токена EOS (End of Sentence)
tokenizer.pad_token = tokenizer.eos_token

# Функция для токенизации данных
def preprocess_function(examples):
    inputs = examples['input']
    outputs = examples['output']
    
    # Склеиваем вход и выход с помощью токенизатора
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(outputs, max_length=512, truncation=True, padding='max_length')
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Применяем функцию предобработки к датасету
tokenized_dataset = dataset.map(preprocess_function, batched=True)


# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(model_name)

# Если вы хотите использовать метод LoRA (Low-Rank Adaptation) для экономии ресурсов:
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # ранг матрицы
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # целевые слои
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="crystall_generation/finetune/results/",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_dir='crystall_generation/finetune/logs/',
    logging_steps=10,
    fp16=True,  # Использование mixed precision для ускорения
    push_to_hub=False,  # Если вы хотите загрузить модель на Hugging Face Hub
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Запуск обучения
trainer.train()

from transformers import pipeline
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoTokenizer

# Загрузка fine-tuned модели
model = AutoModelForCausalLM.from_pretrained("crystall_generation/models/fine_tuned_Llama-3.2-3b")
tokenizer = AutoTokenizer.from_pretrained("crystall_generation/models/fine_tuned_Llama-3.2-3b")

# Создание пайплайна для генерации текста
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
input_text = "band_gap=0.0 spacegroup.number=139"
output = generator(input_text, max_length=50)
print(output)