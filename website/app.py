import os
import streamlit as st
from transformers import pipeline
import re

os.environ["STREAMLIT_DISABLE_LOCAL_FILES_WATCHER"] = "true"

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Путь к модели '{model_path}' не существует. Проверьте путь.")
    return pipeline("text-generation", model=model_path)

def generate_text(input_text, generator, max_length, temperature, num_return_sequences):
    outputs = generator(
        input_text,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        truncation=False
    )
    return outputs

VALID_ATOMS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
}

def validate_slices(slices_string):
    """
    Валидация строки SLICES и вычисление процента соответствия формату.
    
    Args:
        slices_string (str): Сгенерированная строка SLICES.
    
    Returns:
        dict: Словарь с результатами валидации:
            - 'is_valid' (bool): Соответствует ли строка формату SLICES.
            - 'valid_percentage' (float): Процент соответствия формату SLICES.
            - 'errors' (list): Список ошибок, если они есть.
    """
    result = {
        'is_valid': False,
        'valid_percentage': 0.0,
        'errors': []
    }
    
    parts = slices_string.strip().split()
    total_parts = len(parts)
    valid_parts = 0
    
    for i, part in enumerate(parts):
        if part in VALID_ATOMS or part in {'o', '+', '-'} or part.isdigit():
            valid_parts += 1
        else:
            result['errors'].append(f"Недопустимый символ в позиции {i}: '{part}'")
    
    if total_parts > 0:
        result['valid_percentage'] = (valid_parts / total_parts) * 100
    
    if result['valid_percentage'] >= 90:
        result['is_valid'] = True
    
    return result

def main():
    st.title("Генератор кристаллических структур на базе GPT-2")
    st.write("Введите параметры `band_gap` и `formation_energy`, а затем дополнительный текст.")

    model_path = r"gruz2520/crystall_generation/models/fine_tuned_gpt2_on_alex_full/"

    try:
        generator = load_model(model_path)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    st.subheader("Введите параметры:")
    band_gap = st.number_input("Band Gap", value=0.0, step=0.1, format="%.2f")
    formation_energy = st.number_input("Formation Energy", value=-0.7, step=0.1, format="%.2f")

    st.subheader("Введите дополнительный текст:")
    additional_text = st.text_area("Текст:", height=100)

    prompt = f"{additional_text} {band_gap} {formation_energy} ->"

    st.subheader("Текстовая подсказка:")
    st.write(prompt)

    st.subheader("Настройте параметры генерации:")
    max_length = st.slider("Максимальная длина текста", 10, 200, 50)
    temperature = st.slider("Температура", 0.1, 1.0, 1.0)
    num_return_sequences = st.slider("Количество вариантов", 1, 10, 3)

    if st.button("Сгенерировать"):
        with st.spinner("Генерация текста..."):
            results = generate_text(
                input_text=prompt,
                generator=generator,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences
            )
        st.success("Генерация завершена!")
        st.subheader("Результаты:")

        for i, result in enumerate(results):
            generated_text = result["generated_text"]
            st.write(f"Вариант {i + 1}:")
            st.write(generated_text)

            if "->" in generated_text:
                _, slices_part = generated_text.split("->", 1)
                slices_part = slices_part.strip()
            else:
                slices_part = generated_text.strip()

            validation_result = validate_slices(slices_part)
            st.write("**Показатели валидности:**")
            st.write(f"- Валидность: {'✅ Да' if validation_result['is_valid'] else '❌ Нет'}")
            st.write(f"- Процент соответствия: {validation_result['valid_percentage']:.2f}%")
            if validation_result['errors']:
                st.write("- Ошибки:")
                for error in validation_result['errors']:
                    st.write(f"  - {error}")

if __name__ == "__main__":
    main()