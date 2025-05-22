import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import argparse
from typing import Any, Dict, List

def muat_json(jalur_json: str) -> Dict[str, Any]:
    with open(jalur_json, 'r', encoding='utf-8') as f:
        return json.load(f)

def buat_cell(cell: Dict[str, Any]) -> nbformat.NotebookNode:
    tipe_cell = cell.get('cell_type')
    sumber = cell.get('source', [])
    if isinstance(sumber, str):
        sumber = [sumber]
    if tipe_cell == 'markdown':
        return new_markdown_cell(sumber)
    elif tipe_cell == 'code':
        return new_code_cell(sumber)
    else:
        print(f"Melewati tipe cell tidak dikenal: {tipe_cell}")
        return None

def bangun_notebook(data_cells: List[Dict[str, Any]]) -> nbformat.NotebookNode:
    nb = new_notebook()
    cells = []
    for item in data_cells:
        cell = buat_cell(item)
        if cell:
            cells.append(cell)
    nb['cells'] = cells
    return nb

def simpan_notebook(nb: nbformat.NotebookNode, jalur_output: str) -> None:
    with open(jalur_output, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def json_ke_ipynb(jalur_json: str, jalur_output: str) -> None:
    try:
        data = muat_json(jalur_json)
        nb = bangun_notebook(data.get('cells', []))
        simpan_notebook(nb, jalur_output)
        print(f"Sukses! File .ipynb berhasil disimpan ke: {jalur_output}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

def main():
    parser = argparse.ArgumentParser(description='Konversi JSON ke Jupyter Notebook (.ipynb)')
    parser.add_argument('jalur_json', help='Path file JSON input')
    parser.add_argument('jalur_output', help='Path file .ipynb output')
    args = parser.parse_args()
    json_ke_ipynb(args.jalur_json, args.jalur_output)

if __name__ == "__main__":
    main()
