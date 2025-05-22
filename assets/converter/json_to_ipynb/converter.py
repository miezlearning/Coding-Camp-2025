from typing import Any, Dict, List
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

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
    cells = [cell for item in data_cells if (cell := buat_cell(item))]
    nb['cells'] = cells
    return nb
