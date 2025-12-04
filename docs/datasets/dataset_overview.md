Acest document oferă o prezentare generală a seturilor de date utilizate în proiectul
„Sistem de asistență pentru șofer bazat pe rețele neuronale”.

 1. Tipuri de date

Proiectul utilizează două tipuri principale de dataset-uri:

1. **Dataset vehicule și plăcuțe de înmatriculare**
   - plăcuțe generate sau extrase din imagini;
   - informații sintetice despre șoferi și incidente;
   - folosit pentru modelul de risc al vehiculului.

2. **Dataset intersecții**
   - imagini ale unor intersecții din București (Arcul de Triumf, Piața Romană, Universitate);
   - scoruri sintetice de risc asociate fiecărei intersecții;
   - folosit pentru modelul de clasificare a intersecțiilor.

# Descrierea Setului de Date

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:**  
  Dataset-ul este **sintetic** și a fost realizat special pentru proiect.  
  Conține:
  - plăcuțe de înmatriculare auto generate sintetic;
  - imagini brute cu plăcuțe auto (generate cu model AI);
  - imagini brute ale unor intersecții din București (Arcul de Triumf, Piața Romană, Universitate);
  - date fictive despre vehicule și „șoferi” inventați;
  - incidente rutiere generate procedural.

* **Modul de achiziție:**  
  ☐ Senzori reali  
  ☐ Simulare hardware  
  ☐ Fișier extern  
  ☑ **Generare programatică (Python) + imagini sintetice AI**

* **Perioada / condițiile colectării:**  
  Noiembrie 2024 – Ianuarie 2025  
  Date colectate sub condiții controlate, fără date personale reale.  
  Toate informațiile sunt 100% **inventate** pentru scop educațional.

---

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:**  
  - ~500 vehicule sintetice în baza de date  
  - ~1–3 incidente per vehicul (generare stochastică)  
  - 3 intersecții majore (cu scoruri de risc sintetice)  
  - ~10–20 imagini brute cu plăcuțe

* **Număr de caracteristici (features):**  
  Pentru vehicule: **6–8 features**  
  Pentru intersecții: **1–3 features**

* **Tipuri de date:**  
  ☑ Numerice  
  ☑ Categoriale  
  ☑ Temporale (timestamps sintetice)  
  ☑ Imagini (PNG/JPG)

* **Format fișiere:**  
  ☑ CSV – features extrase pentru rețeaua neuronală  
  ☑ TXT – configurări  
  ☑ JSON – mapare scoruri intersecții  
  ☑ PNG / JPG – imagini plăcuțe + intersecții  
  ☑ SQLite DB – vehicule + incidente  

---

### 2.3 Descrierea fiecărei caracteristici

#### 1. **Pentru vehicule**

| Caracteristică       | Tip        | Unitate | Descriere                                             | Domeniu valori |
|----------------------|------------|---------|-------------------------------------------------------|----------------|
| plate                | text       | –       | plăcuța auto sintetică                               | ex: B123ABC    |
| county               | categorial | –       | județul inventat al înmatriculării                   | {B, IS, CJ…}   |
| total_incidents      | numeric    | număr   | numărul total de incidente                            | 0–10           |
| mean_severity        | numeric    | scor    | severitatea medie a incidentelor                      | 1–20            |
| risk_score           | numeric    | scor    | scor final folosit la modelul MLP                     | 0–100          |

---

#### 2. **Pentru intersecții**

| Caracteristică    | Tip      | Unitate | Descriere                           | Domeniu |
|-------------------|----------|---------|-------------------------------------|---------|
| img_path          | imagine  | PNG     | imagine brută a intersecției        | –       |
| location_label    | text     | –       | denumirea intersecției              | {AT, PR, UNIV} |
| accident_score    | numeric  | scor    | scor sintetic risc (ex: 85/70/60)   | 0–100   |
| class_label       | text     | –       | clasificare risc                     | {low, med, high} |

---

## 3. Analiza Exploratorie a Datelor (EDA)

### 3.1 Statistici descriptive aplicate

* Medie / mediană / deviație standard pentru:
  - număr incidente
  - severitate incident
  - scor risc vehicul
* Distribuții:
  - histogramă pentru scor risc
  - histogramă incidente
* Boxplot pentru outlieri
* Matrice de corelație pentru features numerice

Exemple rezultate sintetice:
- scor_risc mediu ≈ 54.3  
- total_incidents mediu ≈ 2.1  
- mean_severity medie ≈ 2.8  

---

### 3.2 Analiza calității datelor

* Valori lipsă: 0% (date generate programatic)
* Inconsistențe: unele plăcuțe pot avea litere rare → normalizare
* Corelații puternice:
  - total_incidents ↔ risk_score
  - mean_severity ↔ risk_score

---

### 3.3 Probleme identificate

* Distribuție **neuniformă** a incidentelor (majoritatea vehiculelor au 0–2 incidente)
* Număr mic de clase pentru intersecții (doar 3)
* Imagini brute cu variații de rezoluție (trebuie redimensionate)

---

## 4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* Eliminare duplicate (rare)
* Normalizare plăcuțe (regex: B123ABC)
* Tratare outlieri pentru scoruri
* Conversie imagini → grayscale / resize

### 4.2 Transformarea caracteristicilor

* Normalizare numerice (Min–Max)
* One-Hot Encoding pentru județ
* Binarizare pentru severe/not severe
* Label encoding pentru intersecții

### 4.3 Structurarea seturilor de date

Recomandare:

* **80% – train**
* **10% – validation**
* **10% – test**

Principii respectate:
* Fără scurgere de informație
* Stratificare în funcție de risc (pentru vehicule)
* Shuffle randomizat

---

### 4.4 Salvarea rezultatelor preprocesării

* Date curate în `data/processed/`
* Seturi pentru rețea:
  - `data/train/`
  - `data/validation/`
  - `data/test/`
* Parametri preprocesare salvați în `config/preprocessing_config.json`

---

## 5. Fișiere Generate în Această Etapă

* `data/raw/` – imagini brute + date necurățate  
* `data/processed/` – imagini transformate, CSV fără anomalii  
* `data/train/`, `data/validation/`, `data/test/` – dataset-uri finale  
* `src/preprocessing/` – codul de preprocesare  
* `data/README.md` – prezentul document  

---

## 6. Stare Etapă

- [x] Structură repository configurată  
- [x] Date brute încărcate (imagini + dataset sintetic)  
- [ ] EDA de implementat în notebook / script  
- [ ] Preprocesare completă  
- [ ] Separare train/val/test  
- [ ] Documentație finalizată  


