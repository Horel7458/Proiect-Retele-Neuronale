# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Petre Horia  
**Link Repository GitHub:** https://github.com/Horel7458/Proiect-Retele-Neuronale.git  
**Data:** 20.01.2026

---

## 0. Rezumat (ce face proiectul)

Proiectul implementează un sistem de estimare a **riscului rutier** în intervalul $[0,1]$, folosind:

- **Achiziție**: cameră + OCR (EasyOCR) pentru citirea numărului de înmatriculare.
- **Date**: statistici istorice de accidente pentru:
  - intersecție + interval orar;
  - vehicul (accidente asociate numărului);
  - județ (scor mediu de accidente pe județ).
- **Model RN**: un MLP (PyTorch) care primește 3 feature-uri și prezice un **scor de risc** normalizat.
- **UI**: aplicație desktop (Tkinter) care afișează scorul și categoria LOW/MEDIUM/HIGH.

În Etapa 6 am făcut analiza de performanță a modelului, am rulat experimente de optimizare și am tras concluzii finale legate de calitate, limitări și direcții de îmbunătățire.

---

## 1. Artefacte relevante (fișiere proiect)

### Date

- Dataset RN: `data/processed/nn_dataset.csv`
  - coloane: `accidente_intersectie`, `accidente_vehicul`, `scor_judet`, `label_risk`
  - generare: `src/preprocessing/dataset_builder.py`

### Model

- Model antrenat (PyTorch state_dict): `data/processed/model.pth`
- Scaler min-max folosit de UI: `data/processed/nn_scaler.json`

### Scripturi (Etapa 6)

- Evaluare (metrici + top erori): `tools/eval_model.py`
- Experimente de optimizare (baseline + 4 variații): `tools/run_optimization_experiments.py`
- Learning curve (MSE vs mărimea setului de antrenare): `src/neural_network/learning_curve.py`
- Output experimente (tabel complet): `data/processed/optimization_experiments.csv`
- Output learning curve: `data/processed/learning_curve.csv` (+ `learning_curve.png` dacă există matplotlib)

---

## 2. Cerințe Etapa 6 (adaptate la proiect)

În proiectul meu ieșirea RN este un **scor continuu** (regresie) în $[0,1]$, deci metrica principală este MSE/MAE (nu Accuracy/F1 ca în clasificare).

Checklist implementat:

1. Minimum 4 experimente de optimizare (hiperparametri): ✅ (5 experimente rulate)
2. Tabel comparativ experimente cu metrici și observații: ✅
3. Analiza performanței pe test set (MSE/MAE/RMSE/R²): ✅
4. Confusion matrix pe categorii LOW/MEDIUM/HIGH (derivate din scor): ✅
5. Analiza a minim 5 cazuri cu eroare mare: ✅
6. Concluzii, limitări, direcții viitoare: ✅

---

## 3. Analiza performanței modelului (model.pth)

### 3.1 Setup evaluare

- Split: 70% train / 15% val / 15% test (seed=42)
- Normalizare: min-max fit pe train, aplicat pe val/test
- Model: MLP 3-16-8-1, activări ReLU, ieșire Sigmoid
- Script folosit: `tools/eval_model.py`

Dimensiuni split obținute:

- Train: 672
- Val: 144
- Test: 144

### 3.2 Metrici (regresie)

Rezultate (rulate pe 10.02.2026):

| Split | MSE | MAE | RMSE | R² |
|------:|-----:|-----:|------:|-----:|
| Train | 0.00008623 | 0.006163 | 0.009286 | 0.998176 |
| Val   | 0.00010809 | 0.006664 | 0.010397 | 0.997794 |
| Test  | 0.00015000 | 0.007381 | 0.012247 | 0.996753 |

Interpretare:

- Modelul aproximează foarte bine eticheta `label_risk`.
- Performanța ridicată este așteptată deoarece `label_risk` este calculată euristic din aceleași 3 feature-uri (modelul învață în mare această relație).

### 3.3 Confusion matrix pe categorii (LOW/MEDIUM/HIGH)

În UI scorul este mapat în categorii:

- LOW: scor < 0.40
- MEDIUM: 0.40 ≤ scor < 0.70
- HIGH: scor ≥ 0.70

Confusion matrix (test set):

| True \ Pred | LOW | MEDIUM | HIGH |
|-----------:|----:|-------:|-----:|
| LOW        | 63  | 0      | 0    |
| MEDIUM     | 1   | 66     | 1    |
| HIGH       | 0   | 0      | 13   |

Observație: pentru pragurile curente, erorile sunt suficient de mici încât predicțiile nu schimbă categoria pe test set.

### 3.4 Learning curve (bias/variance)

Pentru a înțelege dacă modelul este limitat de **bias** (underfitting) sau de **variance** (overfitting), am rulat un learning curve pe fracții din setul de antrenare folosind `src/neural_network/learning_curve.py`.

Rezultate (din `data/processed/learning_curve.csv`):

| Train frac | Train size | Train MSE | Val MSE |
|----------:|-----------:|----------:|--------:|
| 0.10 | 67  | 0.00035271 | 0.00068513 |
| 0.20 | 134 | 0.00022032 | 0.00049352 |
| 0.40 | 268 | 0.00021077 | 0.00029765 |
| 0.60 | 403 | 0.00013365 | 0.00023978 |
| 0.80 | 537 | 0.00010289 | 0.00018463 |
| 1.00 | 672 | 0.00005941 | 0.00013594 |

Interpretare:

- Pe măsură ce crește setul de train, **train MSE scade** rapid și apoi se stabilizează.
- **Val MSE scade** constant și rămâne mai mare decât train MSE → există o mică diferență train/val, dar nu una dramatică.
- Concluzie: modelul nu pare sever overfit; performanța ridicată e în principal explicată de faptul că `label_risk` este o funcție euristică din aceleași 3 feature-uri.

---

## 4. Analiza a 5 exemple cu eroare mare (test set)

Cele mai mari 5 erori absolute (din `tools/eval_model.py`):

| # | Plate | Intersecție | Interval | y_true | y_pred | abs_err | True cat | Pred cat |
|---:|------|------------|----------|-------:|-------:|--------:|----------|----------|
| 1 | AG444POV | Piața Universității | Seara | 1.000000 | 0.915944 | 0.084056 | HIGH | HIGH |
| 2 | AG444POV | Arcul de Triumf | Dimineața | 0.956522 | 0.902091 | 0.054431 | HIGH | HIGH |
| 3 | AR06XAA  | Piața Romană | Prânz | 0.021739 | 0.071737 | 0.049997 | LOW | LOW |
| 4 | AB666JEB | Piața Romană | Dimineața | 0.255964 | 0.225426 | 0.030538 | LOW | LOW |
| 5 | BN567BNM | Piața Unirii | Dimineața | 0.294203 | 0.272982 | 0.021221 | LOW | LOW |

Observații și cauze probabile:

1. **Saturație aproape de 1.0**: Sigmoid comprimă extremele, deci valorile foarte mari sunt mai greu de potrivit exact.
2. **Scoruri foarte mici**: la capătul inferior apar erori relative mai mari (deși abs_err rămâne mic).
3. **Nu există schimbări de categorie**: pragurile sunt suficient de late încât erorile să nu traverseze LOW/MEDIUM/HIGH.

Soluții posibile (dacă aș continua):

- Ieșire liniară + clipping sau calibrare post-hoc pentru extreme.
- Redefinirea label-ului (ex: sigmoid pe combinația brută, ca în `nn_dataset_labeled.csv`).
- Adăugarea de feature-uri (ziua săptămânii, meteo, trafic) și validare temporală.

---

## 5. Optimizare hiperparametri (minimum 4 experimente)

Am rulat 5 experimente (Baseline + 4 variații) folosind `tools/run_optimization_experiments.py`.

### 5.1 Tabel experimente (rezumat)

| Exp | Arhitectură | LR | Dropout | Test MSE | Test MAE | Test R² | Timp (sec) | Observații |
|-----|------------|----:|--------:|---------:|---------:|--------:|-----------:|------------|
| Baseline | 3-16-8-1 | 0.003 | 0.0 | 0.00007933 | 0.004085 | 0.998147 | 1.88 | Cel mai bun; reproduce bine label-ul euristic |
| Exp1_lr_0.001 | 3-16-8-1 | 0.001 | 0.0 | 0.00029860 | 0.011085 | 0.993026 | 1.97 | Învățare prea lentă → underfitting |
| Exp2_lr_0.01 | 3-16-8-1 | 0.01 | 0.0 | 0.00013783 | 0.006077 | 0.996781 | 2.01 | LR prea mare → convergență mai slabă |
| Exp3_bigger_net | 3-32-16-1 | 0.003 | 0.0 | 0.00017760 | 0.007598 | 0.995852 | 2.50 | Capacitate mai mare, dar nu ajută pe date simple |
| Exp4_dropout_0.2 | 3-16-8-1 | 0.003 | 0.2 | 0.00027603 | 0.010919 | 0.993553 | 2.14 | Dropout reduce performanța (problemă deja simplă) |

Rezultatul complet este salvat în `data/processed/optimization_experiments.csv`.

### 5.2 Configurația finală aleasă

Modelul final rămâne **Baseline**, deoarece are cel mai mic Test MSE.

---

## 6. Integrare aplicație (impactul Etapei 6)

În această etapă am verificat că pipeline-ul end-to-end funcționează:

1. OCR → extragere număr → matching în CSV
2. Extragere feature-uri (intersecție, vehicul, județ)
3. Normalizare cu `nn_scaler.json`
4. Inferență RN cu `model.pth`
5. Afișare scor + categorie LOW/MEDIUM/HIGH

Notă: există încă path-uri absolute în unele fișiere. Pentru o versiune complet portabilă, următorul pas ar fi trecerea pe path-uri relative (ca în `src/preprocessing/dataset_builder.py`).

Update: scripturile de antrenare/evaluare folosesc acum path-uri relative la rădăcina proiectului (detectată automat prin folderul `data/`), pentru rulare reproductibilă după clonare.

---

## 7. Concluzii finale

1. Modelul RN rezolvă corect problema definită pe datele actuale: R² ~ 0.998 pe test.
2. Optimizarea arată că baseline-ul este suficient; complexitatea suplimentară nu ajută.
3. Cea mai mare limitare este `label_risk`: nu este ground-truth real, ci o formulă euristică.

Direcții viitoare:

- Date reale + redefinire label.
- Extindere feature-uri și validare temporală.
- Calibrare praguri LOW/MEDIUM/HIGH pe costuri FN/FP relevante utilizatorului.

---

## 8. Instrucțiuni de rulare (Etapa 6)

### Evaluare (metrici + top erori)

```powershell
cd (git rev-parse --show-toplevel)
./.venv/Scripts/python.exe tools/eval_model.py
```

### Experimente optimizare (tabel comparativ)

```powershell
cd (git rev-parse --show-toplevel)
./.venv/Scripts/python.exe tools/run_optimization_experiments.py
```

### Learning curve

```powershell
cd (git rev-parse --show-toplevel)
./.venv/Scripts/python.exe src/neural_network/learning_curve.py
```

### Rulare UI

Conform README.md (UI principal):

```powershell
./.venv/Scripts/python.exe generarenumere/ui_app.py
```
