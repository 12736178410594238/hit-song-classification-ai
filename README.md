# Hit Song Classification AI  
히트곡 판별 머신러닝 프로젝트

---

## 1. Project Overview | 프로젝트 개요

This project builds a machine learning model to classify whether a song is a hit using Spotify audio features.  
본 프로젝트는 Spotify 데이터셋을 기반으로 곡의 특성을 분석하여 히트곡 여부를 예측하는 머신러닝 모델을 구축합니다.

Key Question:
- What makes a song a hit?

---

## 2. Objective | 목표

- Analyze key audio features
- Compare hit vs non-hit songs
- Build binary classification model
- Extract meaningful insights

---

## 3. Dataset | 데이터셋

- Spotify Tracks Dataset
- ~114,000 songs

### Features

| Feature | Description |
|--------|-------------|
| popularity | popularity score |
| danceability | dance suitability |
| energy | intensity |
| valence | positivity |
| tempo | BPM |
| duration_ms | length |
| track_genre | genre |

---

## 4. Problem Definition | 문제 정의

Binary Classification

```python
df["hit"] = df["popularity"].apply(lambda x: 1 if x >= 70 else 0)
```

---

## 5. Preprocessing | 전처리

```python
df = df[df["popularity"] <= 100]
df = df[df["tempo"] >= 0]
df = df[df["duration_ms"] >= 0]
```

- remove invalid values
- feature selection
- target creation

---

## 6. EDA | 탐색적 분석

Methods:
- Box Plot
- Violin Plot
- Heatmap
- Scatter + Trend

Insights:
- energy ↑ → hit 가능성 ↑
- danceability ↑ → hit 가능성 ↑
- tempo 영향 낮음
- 패턴 기반 접근 중요

---

## 7. Modeling | 모델링

Models:
- Logistic Regression
- Random Forest

```python
features = ["danceability", "energy", "valence", "tempo", "duration_ms"]
```

---

## 8. Results | 결과

- single feature로 설명 어려움
- multiple feature 조합 중요
- 데이터 기반 패턴 존재

---

## 9. Limitations | 한계

- popularity 기준 단일 정의
- 외부 요인 미반영
- 메타데이터 기반 분석

---

## 10. Future Work | 향후 개선

- XGBoost / LightGBM
- 장르별 모델
- 오디오 데이터 활용
- Explainable AI

---

## 11. Tech Stack

- Python
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-learn

---

## 12. Project Structure

```
hit-song-classification-ai/
├── data/
├── notebooks/
├── src/
├── outputs/
├── README.md
└── requirements.txt
```

---

## 13. Commit Strategy

```
init: initial setup
prep: preprocessing
eda: visualization
model: training
eval: evaluation
docs: documentation
```

---

## 14. What I Learned

- preprocessing importance
- visualization insights
- pattern-based thinking
- ML workflow understanding

---

## 15. Author

서용준  
AI / ML Portfolio
