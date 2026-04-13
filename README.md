# Hit Song Classification AI

A machine learning project for classifying whether a song is likely to be a hit using Spotify audio features.  
This repository covers the full workflow from data preprocessing and exploratory data analysis to binary classification modeling and interpretation.

---

## 1. Project Overview

In the music industry, only a small portion of released tracks achieve strong public traction.  
This project explores whether a song's measurable audio features can be used to distinguish hit songs from non-hit songs.

The project was designed as an end-to-end machine learning workflow using structured music metadata.  
It includes:

- data preprocessing
- exploratory data analysis (EDA)
- feature selection
- binary classification modeling
- result interpretation

---

## 2. Objective

The goals of this project are as follows:

- identify key features associated with hit songs
- analyze differences between hit songs and non-hit songs through visualization
- build a binary classification model for hit prediction
- derive interpretable insights from model outputs and feature relationships

---

## 3. Dataset

- **Source**: Spotify Tracks Dataset
- **Scale**: approximately 114,000 tracks
- **Type**: tabular dataset

### Main Features

| Feature | Description |
|---------|-------------|
| popularity | Popularity score of a track (0-100) |
| danceability | How suitable a track is for dancing |
| energy | Intensity and activity level of a track |
| valence | Musical positivity conveyed by a track |
| tempo | Estimated tempo in beats per minute (BPM) |
| duration_ms | Track duration in milliseconds |
| track_genre | Genre label of the track |

---

## 4. Problem Definition

This project defines hit prediction as a **binary classification** task.

- `hit = 1`: hit song
- `hit = 0`: non-hit song

The target variable is derived from the `popularity` field.

```python
df["hit"] = df["popularity"].apply(lambda x: 1 if x >= 70 else 0)
```

This threshold is a practical project-level definition based on the dataset's popularity metric.  
It does not directly reflect external factors such as marketing, release timing, social media virality, or artist influence.

---

## 5. Data Preprocessing

Before analysis and modeling, the dataset was cleaned and filtered to improve consistency.

### Preprocessing Steps

- checked basic structure and data types
- inspected missing values
- filtered invalid numeric ranges
- reviewed outliers in core variables
- created the derived target column `hit`

### Example Filtering Logic

```python
# keep valid popularity values
df = df[df["popularity"] <= 100]

# remove negative tempo values
df = df[df["tempo"] >= 0]

# remove negative duration values
df = df[df["duration_ms"] >= 0]
```

### Selected Features for Modeling

The following features were selected as core predictors:

- `danceability`
- `energy`
- `valence`
- `tempo`
- `duration_ms`

These variables were chosen because they provide interpretable musical characteristics and showed practical value during EDA.

---

## 6. Exploratory Data Analysis

EDA was conducted to compare the distribution of major features and inspect relationships between variables.

### Visualization Methods

- **Box Plot**: distribution and outlier inspection
- **Violin Plot**: density and distribution comparison
- **Heatmap**: correlation analysis
- **Scatter Plot with Trend Line**: pairwise relationship analysis

### Key Observations

- hit songs tend to show relatively higher `energy` and `danceability`
- `tempo` does not show a strong standalone linear relationship with hit status
- individual variables alone do not fully explain hit outcomes
- combinations of multiple features appear more informative than any single feature

---

## 7. Modeling

### Task Type
- Binary Classification

### Models
- Logistic Regression
- Random Forest

### Input Features

```python
features = ["danceability", "energy", "valence", "tempo", "duration_ms"]
X = df[features]
y = df["hit"]
```

### Workflow

1. split the dataset into training and test sets
2. train baseline and tree-based models
3. evaluate predictions
4. compare model behavior
5. interpret the results

---

## 8. Results

The modeling results suggest that hit prediction cannot be explained by one variable alone.  
However, a useful classification pattern emerges when multiple musical features are considered together.

### Summary of Findings

- no single feature is sufficient to define a hit song
- combined feature patterns improve predictive usefulness
- energetic and danceable tracks are more likely to align with hit characteristics
- structured music metadata can partially capture hit-song tendencies

This project focuses not only on prediction performance, but also on understanding how data-driven patterns relate to song popularity.

---

## 9. Limitations

This project has several limitations:

- hit status is defined only by the `popularity` variable
- external factors such as artist fame, release strategy, trend cycles, and SNS virality are not included
- analysis is based on metadata, not raw audio signals
- genre-specific differences may require more granular modeling

---

## 10. Future Work

Potential extensions include:

- comparing additional models such as XGBoost or LightGBM
- building genre-specific classifiers
- incorporating raw audio features such as MFCC or spectrogram-based inputs
- integrating external trend or temporal data
- applying explainable AI methods for deeper model interpretation

---

## 11. Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Environment**: Jupyter Notebook, VS Code

---

## 12. Project Structure

```bash
hit-song-classification-ai/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── outputs/
│   ├── figures/
│   └── metrics/
├── README.md
└── requirements.txt
```

---

## 13. Commit Strategy

For a portfolio project, commit history should show a clear and logical development process.

### Suggested Commit Prefixes

- `init:` initial repository setup
- `data:` dataset loading and initial setup
- `prep:` preprocessing and target engineering
- `eda:` exploratory data analysis and visualization
- `model:` model training and tuning
- `eval:` evaluation and result summarization
- `docs:` README and documentation updates
- `refactor:` codebase cleanup and restructuring
- `fix:` bug fixes

### Example Commit Flow

```bash
init: initialize repository with README and project structure
data: add dataset loading and inspection notebook
prep: clean core numeric features and create hit target
eda: add distribution plots for selected audio features
eda: add heatmap and scatter plot analysis
model: train logistic regression baseline
model: train random forest classifier
eval: add evaluation metrics and confusion matrix
refactor: separate preprocessing and training modules
docs: refine README for portfolio presentation
```

---

## 14. What I Learned

Through this project, I practiced the complete workflow of a machine learning classification task using structured data.

Key takeaways include:

- the importance of preprocessing and feature selection
- the role of visualization in understanding model inputs
- the limitations of relying on a single popularity-based target
- the value of interpreting model behavior, not just optimizing performance

---

## 15. Author

**Yongjun Seo**  
AI / Machine Learning Portfolio Project
