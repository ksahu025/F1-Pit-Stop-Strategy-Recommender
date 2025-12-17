# F1 Pit Stop Lap and Count Prediction Using Machine Learning


## Project Overview

Formula 1 racing is one of the most strategically complex sports in the world. Among the many tactical decisions, **pit stop strategy**—the timing and frequency of stops—can determine the difference between victory and defeat.

This project aims to build a **data-driven pit stop strategy recommender system** that answers two key questions:
1.  **Lap Prediction:** *When* will a pit stop occur? (Regression)
2.  **Strategy Classification:** *How many* stops will a driver make? (1-stop, 2-stop, or 3+ stops)

A major focus of this project is analyzing **Distribution Shift** in data and strategy: How machine learning models adapt (or fail to adapt) when regulations change significantly, specifically analyzing the impact of the **2022 F1 regulation overhaul**. Using this data to predict subsequent pit stops and pit lap count.

---

**Course:** DATA602: Principles of Data Science (Fall 2025)    

## Team Members

| Name | ID |
| :--- | :--- |
| **Stuti Patel** | 122013834 |
| **Riya Puri** | 122099296 |
| **Kshitij Lnu** | 121507990 |

---

## Data Curation & Pipeline

**Source:** [Ergast Developer API](http://ergast.com/mrd/) (via Kaggle).  
**Scope:** Races from **2016 to 2024** to ensure relevance to modern electronic timing and racing conditions.

### 1. Data Processing
We integrated 7 separate CSV files to create a master dataset:
* `races.csv`, `results.csv`, `pit_stops.csv` (Target variables)
* `circuits.csv`, `drivers.csv`, `constructors.csv` (Contextual metadata)
* `qualifying.csv` (Performance metrics)

**Key Cleaning Steps:**
* **Temporal Filtering:** Removed pre-2016 data to reduce noise. Major regulation changes in *2014–2015* (due to hybrid engines etc.),  which significantly dropped the average number of pitstops.
* **Imputation:** Handled missing qualifying positions (e.g., pit lane starts) by defaulting to grid position 20.
* **Filtering:** Removed "Did Not Finish" (DNF) records (<90% race distance) to avoid skewed stop counts.

### 2. Feature Engineering
We generated **38 features** focusing on "Time-Awareness" to prevent data leakage:
* **Recent Rolling Averages:** 10, 15, and 20-race windows for drivers, constructors, and circuits.
* **Historical Averages:** Expanding window averages for long-term trends.
* **Temporal Flags:** `is_2022_plus` binary flag to capture the regulation era shift.
* **Context:** Grid position, track latitude/longitude, and season progress percentage.

---

## Methodology & Models

We formulated the problem into two distinct tasks:

### Task A: Lap Number Prediction (Regression)
* **Algorithm:** Random Forest Regressor
* **Hyperparameters:** `n_estimators=100`, `max_depth=8`
* **Goal:** Predict the specific lap number (1-70) a driver will box.

### Task B: Strategy Classification (Classification)
* **Algorithm:** XGBoost Classifier
* **Hyperparameters:** `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `reg_alpha=0.1`
* **Why XGBoost?** Chosen for its ability to handle **class imbalance** and **distribution shifts** better than Random Forest via sequential error correction.

### Experimental Design: Chronological vs. Random Split
To simulate real-world deployment, we evaluated models using two splitting strategies for both Regressions and Classification models:
1.  **Chronological Split (Realistic):** Train (2016-Mid 2022) / Validation / Test (Late 2023-2024). We do not want to train our model on future data to make past data predictions.
2.  **Random Split (Optimistic):** Shuffled data, which ignores temporal evolution.

---

## Key Results & Findings

### 1. The "2022 Shift"
Our EDA revealed a fundamental shift in strategy following the 2022 regulations:
* **Pre-2022:** 1-stop strategies were dominant (44%).
* **Post-2022:** 2-stop strategies became dominant (46%).
* *Impact:* Models trained solely on old data struggle to predict the aggressive 2-stop strategies of the current era.

### 2. Relevance of removing data before 2016

- ⁠Major regulation changes in *2014–2015* (hybrid engines) causing changes in strategy and overall stats
- Different tire behavior and degradation patterns
- ⁠Different pit stop norms and race strategies. The average number pit stops per driver reduces significantly after 2015 regulation change.
- Smaller and less consistent pit stop datasets

### 3. Model Performance

| Model | Task | Algorithm | Split | Train Score | Test Score |
|-------|------|-----------|-------|-------------|------------|
| A1 | Lap Number Regression | RandomForestRegressor | Chronological 70/15/15 | R2: 0.667 | R2: 0.313 |
| A2 | Pit Stop Classification | **XGBClassifier** | Chronological 85/15 | Acc: 78.1 | Acc: 47.5 |
| B1 | Lap Number Regression | RandomForestRegressor | Random 70/15/15 | R2: 0.644 | R2: 0.565 |
| B2 | Pit Stop Classification | **XGBClassifier** | Random 70/15/15 | Acc: 79.0 | Acc: 64.4 |

### 3. Feature Importance
Recent data is more valuable than deep history. The top featurees for our prediction were:
1.  `circuit_recent_avg` (Rolling average of stops at this track)
2.  `constructor_recent_avg` (Team's recent strategic tendencies)
3.  `total_race_laps`

---

## References
1. **Kaggle F1 Dataset:** https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020 
2. [cite_start]**Data Source:** [Ergast Developer API](http://ergast.com/mrd/) [cite: 317]
2.  **XGBoost:** Chen, T., & Guestrin, C. (2016). [cite_start]*XGBoost: A Scalable Tree Boosting System.* [cite: 320]
3.  **Random Forest:** Breiman, L. (2001). [cite_start]*Random Forests.* Machine Learning. [cite: 321]
4. **Formula 1 Official Website:** https://www.formula1.com/

---

## Repository Structure

```bash
├── data/                          # Raw CSV datasets sourced from Ergast API
│   ├── races.csv                  # Race metadata (date, circuit, round)
│   ├── results.csv                # Race results per driver
│   ├── pit_stops.csv              # Individual pit stop records (Target)
│   ├── circuits.csv               # Circuit information (location, altitude)
│   ├── drivers.csv                # Driver information
│   ├── constructors.csv           # Team/constructor information
│   └── qualifying.csv             # Qualifying session results
├── f1_pitstop_prediction.ipynb    # Main Jupyter Notebook with code & analysis
├── index.html                     # Comprehensive Project Report and Tutorial
└── README.md                      # Project documentation