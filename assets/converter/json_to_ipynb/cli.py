import argparse
from .io_utils import muat_json, simpan_notebook
from .converter import bangun_notebook

def json_ke_ipynb(jalur_json: str, jalur_output: str) -> None:
    try:
        data = muat_json(jalur_json)
        nb = bangun_notebook(data.get('cells', []))
        simpan_notebook(nb, jalur_output)
        print(f"Sukses! File .ipynb berhasil disimpan ke: {jalur_output}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

def run_cli():
    parser = argparse.ArgumentParser(description='Konversi JSON ke Jupyter Notebook (.ipynb)')
    parser.add_argument('jalur_json', help='Path file JSON input')
    parser.add_argument('jalur_output', help='Path file .ipynb output')
    args = parser.parse_args()
    json_ke_ipynb(args.jalur_json, args.jalur_output)
