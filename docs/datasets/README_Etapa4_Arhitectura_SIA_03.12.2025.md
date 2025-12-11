# Etapa 4 – Arhitectura Sistemului Inteligent Autonom (SIA)
**Disciplina:** Rețele Neuronale  
**Universitatea:** Universitatea POLITEHNICA din București – FIIR  
**Student:** *[Petre Horia]*  
**Grupa:** *[632AB]*  
**Repo GitHub:** *[https://github.com/Horel7458/Proiect-Retele-Neuronale.git]*  
**Data:** *[04-12-2025]*  

---

# 1. Introducere și scopul aplicației

Proiectul urmărește dezvoltarea unui **Sistem Inteligent Autonom (SIA)** capabil să analizeze riscul rutier pe baza a două surse principale de informații:

1. **Numere de înmatriculare**, recunoscute în timp real prin OCR.  
2. **Context rutier**, reprezentat de:
   - intersecția selectată de utilizator  
   - intervalul orar curent  
   - istoricul de accidente asociat intersecției
   - istoricul de accidente asociat vehiculului  

Sistemul este conceput astfel încât să ofere:

- detectarea automată a plăcuței auto  
- evaluarea dacă vehiculul prezintă risc (istoric de accidente)  
- estimarea riscului intersecției în funcție de ora zilei  
- pregătirea input-urilor pentru antrenarea unei rețele neuronale  
- o interfață complet funcțională pentru utilizator  

**În Etapa 4 modelul neuronal nu este antrenat**, dar este complet definit și integrat în arhitectură.

---

# 2. Tabel „Nevoie Reală → Soluție SIA → Modul Software”

Acest tabel reflectă legătura dintre o problemă reală, soluția inteligentă propusă și modulul software responsabil.

| **Nevoie reală concretă** | **Soluția SIA** | **Modul responsabil** |
|---------------------------|-----------------|------------------------|
| Identificarea rapidă a unei mașini cu istoric problematic | Sistem OCR recunoaște plăcuța → verifică în baza de date → afișează nr. accidente | OCR Engine + CSV Data Lookup |
| Estimarea riscului într-o intersecție la momentul curent | Folosirea datelor istorice din CSV + model  pentru estimare risc | Intersection Data Module + RN Model |
| Prezentarea rapidă a informațiilor șoferului | UI care integrează OCR, baza de date, selecția intersecției și afișarea rezultatelor | Tkinter UI |
| Pregătirea datelor pentru antrenarea rețelelor neuronale | Preprocesare, encoding numeric intersecție + interval | Preprocessing Module |

---

# 3. Contribuția Originală la Dataset (cerință ≥ 40%)

Pentru acest proiect, dataset-ul este **100% creat de mine**, depășind cerința minimă.

## 3.1 Tipuri de date generate

### ✔ Date generate complet original:
- Plăcuțele de înmatriculare (cca. 100)  
- Numărul de accidente asociat fiecărei plăcuțe  
- CSV complet pentru intersecții:
  - 4 intersecții
  - 3 intervale orare
  - număr de accidente realist distribuit pe intervale

### ✔ Date achiziționate prin senzori:
- imagini reale capturate cu camera laptopului  
- folosite pentru recunoașterea OCR

### ✔ Date sintetice pentru antrenare ulterioară:
- distribuții de accidente pe intervale  
- mapări intersecție → risc

## 3.2 Structura dataset-ului

- `data/raw/plates_export.csv` – 100 plăcuțe + accidente  
- `data/raw/intersections.csv` – 12 mostre (4×3 combinații)  
- `data/test/` – capturi OCR  
- `data/generated/` – imagini și date sintetice  

Total: **144 mostre validate**, toate originale.

---

# 4. Arhitectura Sistemului (State Machine)

SIA este construit ca o **mașină de stări finite (FSM)**.  

## 4.1 Stări definite

1. **IDLE**  
   Sistemul este pornit și așteaptă o acțiune.

2. **ACQUIRE_DATA**  
   Utilizatorul fie:
   - scanează o plăcuță cu camera  
   - selectează intersecția și intervalul orar  

3. **PREPROCESS**  
   - curățare text OCR  
   - verificare în CSV  
   - encodare intersecție + interval → input rețea neuronală  

4. **RN_INFERENCE**  
   Modelul neuronal (definit, neantrenat) produce un scor între 0 și 1.  
   În Etapa 4 scorul este simbolic.

5. **DISPLAY_RESULT**  
   UI afișează rezultatul:
   - numărul detectat  
   - accidente vehicul  
   - accidente intersecție pentru interval  

6. **LOG_RESULT**  
   Acțiunea și rezultatul sunt salvate local.

7. **ERROR**  
   - camera nu funcționează  
   - OCR nu detectează text  
   - CSV lipsește  

## 4.2 Justificare arhitecturală

Structura FSM permite:

- modularitate  
- gestionarea erorilor  
- extinderea ușoară cu noi state  
- integrare naturală a rețelei neuronale în pipeline  

---

# 5. Modulele Software ale Aplicației

## 5.1 Modulul 1 – Data Logging & Acquisition  
**Director:** `src/data_acquisition/`

### Responsabilități:
- generare automată CSV intersecții (`generate_intersections.py`)  
- captură imagini OCR (`capture_plate.py`)  
- colectare date brute pentru input RN  

Acest modul asigură toate datele necesare pentru Etapa 5 (antrenare).

---

## 5.2 Modulul 2 – Neural Network Module  
**Director:** `src/neural_network/model.py`

Modelul propus este un **Multi-Layer Perceptron (MLP)**.

### Input (3 neuroni):
1. intersecție ID (encoded)  
2. interval orar ID  
3. accidente vehicul  

### Output (1 neuron):
- scor risc ∈ `[0, 1]`

