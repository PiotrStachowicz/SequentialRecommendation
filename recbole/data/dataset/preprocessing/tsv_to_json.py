"""
tsv_to_json.py

Provides fast scalable way to convert TSV type files
into JSONL for simpler data analysis using pyarrow.

Author: Piotr Stachwoicz
"""

import csv
import json
import argparse

def retype(row: dict, types: str):
    """Retype the entry"""
    for i, (key, value) in enumerate(row.items(), 0):           
        try:
            if types[i] == "token":
                row[key] = str(value)
            
            if types[i] == "int":
                row[key] = int(value)
            
            if types[i] == "float":
                row[key] = float(value)
            
            if types[i] == "bool":
                row[key] = bool(value)
            
            if types[i] == "token_str":
                row[key] = [str(token) for token in value.split(',')]
        except Exception:
            print('There was an error while converting types!')

    return row


def convert(inp_path: str, out_path: str) -> None:
    """Convert TSV file into JSONL"""
    with (
        open(inp_path, 'r', encoding='utf-8') as inp, 
        open(out_path, 'w+', encoding='utf-8') as out
    ):
        reader = csv.reader(inp, delimiter='\t')

        header = next(reader)
        field_names = [col.split(':')[0] for col in header]
        field_types = [col.split(':')[1] for col in header]

        reader = csv.DictReader(inp, delimiter='\t', fieldnames=field_names)
    

        for row in reader:
            row = retype(row, field_types)
            json.dump(row, out)
            out.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .tsv to .jsonl")
    parser.add_argument("--tsv", required=True, help="Path to input TSV file (.item or .inter)")
    parser.add_argument("--jsonl", required=True, help="Path to output JSONL file")

    args = parser.parse_args()
    convert(args.tsv, args.jsonl)
