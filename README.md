# Vietnamese Restaurant Review Sentiment Analysis

An AI project that analyzes restaurant reviews and shows overall sentiment and aspect-based sentiment.

This project is built as an **end-to-end NLP system**: data → model → prediction → dashboard.

---

## 1. What this project does

* Analyze review sentiment: **Positive / Neutral / Negative**
* Analyze sentiment by aspects:

  * Food
  * Service
  * Price
  * Space
  * Location
* Fine-tune sentiment analysis model (PhoBERT)
* Predict sentiment from new reviews
* Visualize results with a dashboard

---

## 2. Project Workflow

```
Data → Preprocessing → Training → Model → Prediction → Visualization
```

---

## 3. Project Structure

```
viet-restaurant-sas/
│
├── data/               # Raw & processed data
├── notebooks/          # EDA, preprocessing, training, evaluation
├── src/                # preprocess, train, predict
├── models/             # saved model & configs
├── dashboard/          # Streamlit app
├── docs/               # images, demo
│
├── predict_demo.py     # demo prediction
├── requirements.txt
└── README.md
```

---

## 4. Project Goal

The goal of this project is to practice:

* Natural Language Processing (NLP)
* Vietnamese Sentiment Analysis
* PhoBERT fine-tuning
* Building an AI pipeline
* Building a simple analytics dashboard