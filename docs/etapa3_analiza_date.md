# Etapa 3 – Analiza Datelor (EDA) și Pregătire Dataset

Acest document descrie **datele existente în repository** folosite în proiectul „Sistem de asistență pentru șofer bazat pe rețele neuronale”.

În versiunea curentă a proiectului, datele sunt **tabulare (CSV)**, generate sintetic / programatic și nu conțin date personale reale.

## 1. Tipuri de date

Proiectul folosește 3 surse tabulare principale:

1. **Vehicule (plăcuțe + accidente asociate)**
   - fișier: `data/raw/plates_export.csv`
   - proveniență: export din SQLite (`plates.db`) prin `generarenumere/export_csv.py`

2. **Scenarii intersecții (intersecție + interval orar + accidente)**
   - fișier: `data/raw/intersections.csv`
   - proveniență: generare sintetică prin `generarenumere/genereaza_intersections.py`

3. **Statistici pe județe (scor județ)**
   - fișier: `data/processed/stats_by_judet.csv`
   - folosit pentru feature-ul `scor_judet`

Din aceste surse se construiește dataset-ul final pentru RN:

- `data/processed/nn_dataset.csv` (generat de `src/preprocessing/dataset_builder.py`)

## 2. Sursa datelor și originalitate

- **Origine:** date sintetice generate de mine (plăcuțe, accidente pe vehicul, scenarii intersecții) + preprocesare.
- **Perioada generării:** Decembrie 2025 – Februarie 2026.
- **Confidențialitate:** nu sunt folosite date personale reale; plăcuțele sunt generate/introduse sintetic.

## 3. Dimensiuni și structură

### 3.1 Dimensiuni (în acest repo)

- `plates_export.csv`: **80 vehicule**
- `intersections.csv`: **12 scenarii** (combinații intersecție × interval)
- `nn_dataset.csv`: **960 observații** (80 × 12)

### 3.2 Scheme fișiere

**`data/raw/plates_export.csv`**

| Coloană | Tip | Descriere |
|---|---|---|
| `id` | int | ID intern |
| `plate` | str | număr de înmatriculare |
| `accidents` | int | număr accidente asociate vehiculului |

**`data/raw/intersections.csv`**

| Coloană | Tip | Descriere |
|---|---|---|
| `intersection` | str | nume intersecție |
| `interval_label` | str | etichetă interval (ex: Dimineata/Pranz/Seara) |
| `time_range` | str | interval orar (ex: 06:30-09:30) |
| `accidents` | int | număr accidente pentru scenariu (sintetic) |

**`data/processed/stats_by_judet.csv`**

| Coloană | Tip | Descriere |
|---|---|---|
| `county_code` | str | cod județ |
| `numar_vehicule` | int | câte vehicule au fost folosite la calcul |
| `scor_mediu_accidente` | float | scor mediu accidente pe județ |

**`data/processed/nn_dataset.csv`**

| Coloană | Tip | Descriere |
|---|---|---|
| `plate` | str | plăcuță |
| `county_code` | str | cod județ extras din plăcuță |
| `intersection` | str | intersecție |
| `interval_label` | str | interval |
| `time_range` | str | interval orar |
| `accidente_intersectie` | int | accidente scenariu intersecție |
| `accidente_vehicul` | int | accidente vehicul |
| `scor_judet` | float | scor județ |
| `label_risk` | float | etichetă euristică în $[0,1]$ |

## 4. Analiza exploratorie (EDA)

Verificările EDA aplicate pe datele tabulare:

- validare coloane și tipuri (numerice pentru `accidente_intersectie`, `accidente_vehicul`, `scor_judet`)
- lipsă valori: verificare `NaN`/câmpuri goale
- distribuții pentru componentele de risc (vehicul/intersecție/județ)
- corelații așteptate (componente mai mari → `label_risk` mai mare)

Notă: `label_risk` este **calculată euristic** din aceleași 3 feature-uri, deci performanța mare a RN indică învățarea unei funcții construite (nu neapărat predictivitate pe accidente reale).

## 5. Preprocesare și split

Pipeline-ul de construire a dataset-ului (`src/preprocessing/dataset_builder.py`) face:

- extrage `county_code` din `plate`
- face join cu `stats_by_judet.csv` pentru `scor_judet`
- cross-join: vehicule × scenarii intersecții
- normalizează în $[0,1]$ cele 3 componente și calculează `label_risk` ca:

$$
label\_risk = 0.5\cdot acc\_inter\_{01} + 0.3\cdot acc\_veh\_{01} + 0.2\cdot scor\_judet\_{01}
$$

Split-ul pentru train/val/test este realizat în scriptul de antrenare (`src/neural_network/train_nn.py`): **70% / 15% / 15%** (seed=42).

## 6. Fișiere generate / rezultate

- `data/processed/nn_dataset.csv` – dataset final pentru RN
- `data/processed/nn_scaler.json` – scaler Min-Max folosit la antrenare/inferență
- `data/processed/model.pth` – model PyTorch antrenat (etapele ulterioare)

## 7. Stare etapă (în cadrul proiectului final)

- [x] Date brute disponibile în `data/raw/`
- [x] Dataset RN construit în `data/processed/nn_dataset.csv`
- [x] Preprocesare definită (normalizare + etichetare euristică)
