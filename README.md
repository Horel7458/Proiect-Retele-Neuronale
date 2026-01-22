**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Petre Horia]  
**Data:** [11/20/2025]  

## Pornire aplicație (UI)

Aplicația principală pentru interfață este: `generarenumere/ui_app.py`.

### Cerințe

- Python 3.12 (recomandat). Python 3.14 poate să nu aibă roți (wheels) pentru pachete precum `pandas`/`easyocr`.

### Instalare

În PowerShell, din rădăcina proiectului:

1) Creează mediul virtual:

`py -3.12 -m venv .venv`

2) Instalează dependențele:

`./.venv/Scripts/python.exe -m pip install -r requirements.txt`

### Rulare

`./.venv/Scripts/python.exe generarenumere/ui_app.py`
