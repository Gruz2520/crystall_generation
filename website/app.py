import os
import streamlit as st
from transformers import pipeline
import re
import importlib.util

def unpacking_model():
    unpacking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks','scr', 'unpacking.py'))

    spec = importlib.util.spec_from_file_location("unpacking", unpacking_path)
    unpacking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unpacking)

    unpacking.extract_multivolume_archive(r"models/model_arhive/model.7z.001", "models/fine_tuned_gpt2_on_alex_full/")

os.environ["STREAMLIT_DISABLE_LOCAL_FILES_WATCHER"] = "true"

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model path '{model_path}' does not exist. Please check the path.")
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
    Validation of SLICES string and calculation of format compliance percentage.
    
    Args:
        slices_string (str): Generated SLICES string.
    
    Returns:
        dict: Dictionary with validation results:
            - 'is_valid' (bool): Whether the string matches the SLICES format.
            - 'valid_percentage' (float): Percentage of compliance with the SLICES format.
            - 'errors' (list): List of errors, if any.
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
            result['errors'].append(f"Invalid symbol at position {i}: '{part}'")
    
    if total_parts > 0:
        result['valid_percentage'] = (valid_parts / total_parts) * 100
    
    if result['valid_percentage'] >= 90:
        result['is_valid'] = True
    
    return result

def main():
    print("Starting model unpacking")
    unpacking_model()
    print("Model successfully extracted.")
    
    st.title("Crystal Structure Generator Based on GPT-2")
    st.write("Enter `band_gap` and `formation_energy` parameters, then add additional text.")

    model_path = r"models/fine_tuned_gpt2_on_alex_full/"

    try:
        generator = load_model(model_path)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    st.subheader("Enter Parameters:")
    band_gap = st.number_input("Band Gap", value=0.3, step=0.1, format="%.2f")
    formation_energy = st.number_input("Formation Energy", value=-0.7, step=0.1, format="%.2f")

    st.subheader("Enter Additional Text:")
    additional_text = st.text_area("Text:", height=100)

    prompt = f"{additional_text} {band_gap} {formation_energy} ->"

    st.subheader("Text Prompt:")
    st.write(prompt)

    st.subheader("Set Generation Parameters:")
    max_length = st.slider("Maximum Text Length", 10, 200, 50)
    temperature = st.slider("Temperature", 0.1, 1.0, 1.0)
    num_return_sequences = st.slider("Number of Variants", 1, 10, 3)

    if st.button("Generate"):
        with st.spinner("Generating crystal structures..."):
            results = generate_text(
                input_text=prompt,
                generator=generator,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences
            )
        st.success("Generation completed!")
        st.subheader("Results:")

        for i, result in enumerate(results):
            generated_text = result["generated_text"]
            st.write(f"Variant {i + 1}:")
            st.write(generated_text)

            if "->" in generated_text:
                _, slices_part = generated_text.split("->", 1)
                slices_part = slices_part.strip()
            else:
                slices_part = generated_text.strip()

            validation_result = validate_slices(slices_part)
            st.write("**Validation Metrics:**")
            st.write(f"- Validity: {'✅ Yes' if validation_result['is_valid'] else '❌ No'}")
            st.write(f"- Compliance Percentage: {validation_result['valid_percentage']:.2f}%")
            if validation_result['errors']:
                st.write("- Errors:")
                for error in validation_result['errors']:
                    st.write(f"  - {error}")

if __name__ == "__main__":
    main()