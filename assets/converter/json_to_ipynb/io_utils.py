import json
import nbformat
from typing import Any, Dict
from nbformat import NotebookNode

def muat_json(jalur_json: str) -> Dict[str, Any]:
    with open(jalur_json, 'r', encoding='utf-8') as f:
        return json.load(f)

def simpan_notebook(nb: NotebookNode, jalur_output: str) -> None:
    with open(jalur_output, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
