# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Petre Horia]  
**Link Repository GitHub:** [https://github.com/Horel7458/Proiect-Retele-Neuronale.git]  
**Data predării:** [11-12-2025]

---
# Etapa 5 – Antrenarea Rețelei Neuronale (RN)

## 1. Scopul acestei etape

În această etapă se va defini arhitectura, datele de intrare și metoda de antrenare a rețelei neuronale utilizate pentru estimarea riscului rutier.  
Modelul neuronal va folosi informații extrase din:

- numărul de înmatriculare (judetul și istoricul de accidente asociat vehiculului)
- intersecția selectată în aplicație
- intervalul orar curent
- statisticile istorice de accidente pentru fiecare intersecție

RN va avea ca ieșire un **scor de risc normalizat** în intervalul `[0, 1]`, utilizat ulterior în aplicație pentru a avertiza utilizatorul asupra nivelului estimat de risc.

---

## 2. Structura rețelei neuronale (RN)

În această etapă se va utiliza o rețea neuronală de tip **MLP (Multi-Layer Perceptron)**, într-o configurație simplă, potrivită pentru problema abordată.

### **2.1 Arhitectura propusă**

- **Intrări (3–4 features)**:
  - `accidente_intersectie` – număr de accidente din intersecția + intervalul selectat
  - `accidente_vehicul` – numărul de accidente asociat vehiculului
  - `scor_judet` – scor statistic al județului (medie de accidente pe județ)
  
- **Rețea MLP** (fără cod momentan):
  - Layer 1: Fully Connected (4 → 16 neuroni)
  - Activare: ReLU
  - Layer 2: Fully Connected (16 → 8 neuroni)
  - Activare: ReLU
  - Layer final: Fully Connected (8 → 1 neuron)
  - Activare: Sigmoid (pentru ieșire între 0 și 1)

### **2.2 Tipul de rețea**
- Rețea **feed-forward**, fără memorie (nu este recurrentă)
- Potrivită pentru probleme de **regresie** (estimarea unui scor numeric)

---

## 3. Date utilizate pentru antrenare

### **3.1 Structura datasetului**

Modelul folosește în antrenare un set de date derivat din:

- `plates_export.csv`  
  → oferă `accidente_vehicul` și codul de județ  

- `stats_by_judet.csv`  
  → oferă `scor_judet` pentru fiecare județ  

- `intersections.csv`  
  → oferă `accidente_intersectie` pe intersecție + interval orar  

### **3.2 Formatul final al datasetului pentru RN**

Fiecare rând va avea forma:

| feature | descriere |
|--------|-----------|
| accident_intersectie | nr. accidente în intersecția X la intervalul Y |
| accident_vehicul | nr. accidente alocate acelui număr de înmatriculare |
| scor_judet | scorul istoric al județului din prefixul numărului |
| label | risc combinat (0-1), calculat după o formulă euristică inițială |

### **3.3 Normalizarea datelor**
Toate feature-urile numerice vor fi normalizate în `[0, 1]` pentru stabilitatea antrenării.

---

## 4. Metodologie antrenare RN

> **Această secțiune NU conține încă cod**, deoarece antrenarea se va realiza într-o etapă viitoare.

### 4.1 Etapele planificate ale antrenării:

1. **Împărțirea datasetului**

   - 70% antrenare  
   - 15% validare  
   - 15% test  

2. **Funcția de pierdere (Loss Function)**  
   - Recomandată: **MSE (Mean Squared Error)** pentru regresie

3. **Optimizator**  
   - `Adam` cu learning rate între `0.001` și `0.01`

4. **Număr epoci**  
   - 50–200 epoci, în funcție de convergența loss-ului

5. **Evalure model**  
   - MSE pe setul de test  
   - Erori medii pe fiecare tip de scenariu:
     - risc scăzut  
     - risc mediu  
     - risc ridicat  

---

## 5. Integrarea RN în aplicație

După finalizarea antrenării:

1. Modelul va fi exportat (`model.pth` sau `.h5`)
2. Interfața va încărca modelul la pornire
3. La fiecare calcul de risc:
   - se colectează cele 3–4 feature-uri
   - se normalizează la `[0, 1]`
   - se trec prin RN
   - rezultatul este afișat utilizatorului ca:
     - risc numeric
     - categorie risc (low, medium, high)

---

## 6. Limitări și pași următori

### **Limitări curente**
- Modelul nu este încă antrenat
- Nu există rezultate cuantitative
- Formarea datasetului poate necesita extindere pentru robustețe

### **Pași următori**
- Implementarea scriptului de generare dataset pentru RN  
- Implementarea codului de antrenare  
- Analiza metricilor și îmbunătățirea arhitecturii  
- Integrarea modelului în interfață

---

## 7. Concluzii

Această etapă definește cadrul necesar pentru introducerea unui model neuronal predictiv în aplicație.  
Rețeaua neuronală MLP aleasă este potrivită pentru complexitatea proiectului, iar datele existente permit construirea unui model funcțional după generarea datasetului și antrenarea acestuia.

Rezultatele finale vor fi adăugate după efectuarea antrenării în Etapa următoare.

## 8. Structura proiectului

Proiectul *Proiect retele neuronale* este organizat modular, conform cerințelor
de laborator, astfel încât fiecare etapă (achiziție date, preprocesare, antrenare RN)
să fie separată logic și ușor de întreținut.

Structura actuală a proiectului este:

Proiect-Rețele-Neuronale/
│
├── data/
│   ├── raw/
│   │   ├── plates_export.csv          # numere + accidente vehicul
│   │   ├── intersections.csv          # accidente pe intersecții + intervale
│   │   └── intervals.csv              # definirea intervalelor orare
│   │
│   ├── processed/
│   │   └── stats_by_judet.csv         # scoruri medii de risc pe județ
│   │
│   ├── train/                         # pentru antrenarea RN (viitor)
│   ├── test/                          # pentru testarea RN (viitor)
│   └── validation/                    # pentru validare RN (viitor)
│
├── docs/
│   ├── PROIECTLARN.pptx               # prezentarea proiectului
│   ├── state_machine.png               # diagrama FSM
│   │
│   ├── datasets/
│   │   ├── dataset_overview.md
│   │   ├── dataset_overview.txt
│   │   └── README_Etapa4_Arhitectura_SIA_03.12.2025.md
│
├── src/
│   ├── data_acquisition/
│   │   ├── baza_date.py               # crearea bazei de date
│   │   ├── export_csv.py              # export către CSV
│   │   ├── update_accidents.py        # incrementarea accidentelor
│   │   ├── update_plate.py            # actualizare număr înmatriculare
│   │   └── vezi_baza.py               # vizualizare structuri CSV/DB
│   │
│   ├── preprocessing/                 # pregătirea datelor pentru RN (viitor)
│   │
│   ├── neural_network/                # arhitectura + training RN (viitor)
│
├── README.md                          # documentația principală
└── requirements.txt                   # dependențele proiectului


