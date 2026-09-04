def normalize_name(name: str) -> str:
    return name.lower().replace('.', '').replace("'", '').strip()
