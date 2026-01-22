# README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Petre Horia  
**Repository:** https://github.com/Horel7458/Proiect-Retele-Neuronale.git  
**Data:** 20.01.2026

---

## 1. Scopul etapei

În această etapă am configurat și antrenat o Rețea Neuronală (RN) care estimează un **scor de risc rutier** în intervalul $[0,1]$ (problemă de **regresie**).

Features (intrări):

- `accidente_intersectie` – accidente în intersecția selectată + interval orar
- `accidente_vehicul` – accidente asociate vehiculului (număr de înmatriculare)
- `scor_judet` – scor statistic al județului

Ieșirea $[0,1]$ este folosită în UI și mapată în categorii LOW / MEDIUM / HIGH.

---

## 2. Dataset și preprocesare

Surse de date:

- `data/raw/plates_export.csv`
- `data/raw/intersections.csv`
- `data/processed/stats_by_judet.csv`

Dataset final pentru RN:

- `data/processed/nn_dataset.csv` (generat de `src/preprocessing/dataset_builder.py`)
- features: `accidente_intersectie`, `accidente_vehicul`, `scor_judet`
- label: `label_risk` (euristic, în $[0,1]$)

Notă: `label_risk` este derivat euristic din aceleași 3 features, deci performanța mare indică învățarea unei funcții construite, nu neapărat predictivitate pe accidente reale.

---

## 3. Model + antrenare

Model: MLP (PyTorch), 3 → 16 → 8 → 1, activări ReLU, ieșire Sigmoid.

Implementare antrenare: `src/neural_network/train_nn.py`.

Setări (baseline):

- split: 70% train / 15% val / 15% test (seed=42)
- loss: MSE
- optimizer: Adam
- epochs: 120
- batch size: 64

---

## 4. Artefacte generate

- model: `data/processed/model.pth`
- scaler: `data/processed/nn_scaler.json`
- (opțional) curba train/val: `data/processed/train_curve.png`

---

## 5. Rezultate (baseline)

Evaluare realizată cu `tools/eval_model.py`.

Dimensiuni split: Train 664 / Val 142 / Test 142.

| Split | MSE | MAE | RMSE | R² |
|------:|-----:|-----:|------:|-----:|
| Train | 0.00002966 | 0.003497 | 0.005446 | 0.999266 |
| Val   | 0.00002835 | 0.003417 | 0.005324 | 0.999295 |
| Test  | 0.00007933 | 0.004085 | 0.008907 | 0.998147 |

Praguri categorii în UI:

- LOW: scor < 0.40
- MEDIUM: 0.40 ≤ scor < 0.70
- HIGH: scor ≥ 0.70

Pe test set, nu apar schimbări de categorie (confusion matrix perfect diagonal): LOW 66 / MEDIUM 65 / HIGH 11.

---

## 6. Instrucțiuni de rulare (Windows / PowerShell)

Recomandat Python 3.12:

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

./.venv/Scripts/python.exe src/preprocessing/dataset_builder.py
./.venv/Scripts/python.exe src/neural_network/train_nn.py
./.venv/Scripts/python.exe tools/eval_model.py
```

Notă portabilitate: `src/neural_network/train_nn.py` folosește acum path-uri relative la rădăcina proiectului (detectată automat prin folderul `data/`), deci poate rula indiferent unde a fost clonat repository-ul.

---

## 7. Pași următori

- experimente/optimizare hiperparametri + analiză erori (Etapa 6)
- extindere feature set + redefinire label spre o țintă mai realistă
- îmbunătățirea portabilității (eliminare path-uri absolute din unele fișiere)
