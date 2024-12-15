import pandas as pd
import os

def load_csv_tables(base_path, file_names):
    tables = {}
    for file_name in file_names:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            table_name = os.path.splitext(file_name)[0]  # Extract name without extension
            tables[table_name] = pd.read_csv(file_path)
        else:
            print(f"File not found: {file_path}")
    return tables
