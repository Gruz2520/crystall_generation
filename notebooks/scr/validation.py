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

def validate_slices(slices_string, valid_threshold=90):
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
    
    if result['valid_percentage'] >= valid_threshold:
        result['is_valid'] = True
    
    return result