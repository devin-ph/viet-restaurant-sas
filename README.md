<div align="center">

# Vietnamese Restaurant Review ABSA
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![PhoBERT](https://img.shields.io/badge/PhoBERT-vinai%2Fphobert--base-blue?style=flat)
![Gradio](https://img.shields.io/badge/Gradio-Demo-FF7C00?style=flat&logo=gradio&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)

</div>

## About
Fine-tuned PhoBERT model for Aspect-Based Sentiment Analysis (ABSA) on Vietnamese restaurant reviews.  
Given a review, the model predicts which **aspect** is being discussed and the **sentiment** expressed toward it.

🔗 **[Live Demo](https://huggingface.co/spaces/devin-ph/viet-restaurant-absa)**

## Example

| Review | Aspect | Sentiment |
|--------|--------|-----------|
| "Đồ ăn ngon, tôi rất thích" | FOOD#QUALITY | Positive |
| "Phục vụ tệ, thái độ xấu" | SERVICE#GENERAL | Negative |
| "Giá hơi cao so với chất lượng" | FOOD#PRICES | Neutral |

## Dataset

- **Source:** [VLSP2018-ABSA-Restaurant](https://huggingface.co/datasets/visolex/VLSP2018-ABSA-Restaurant)
- **Size:** Over 15k samples after long-format conversion
- **Aspects:** 12 categories (FOOD#QUALITY, SERVICE#GENERAL, AMBIENCE#GENERAL, ...)
- **Sentiment:** 3 classes — positive, neutral, negative
- **Split:** 70% train / 10% val / 20% test

## Model

- **Base:** [vinai/phobert-base](https://huggingface.co/vinai/phobert-base)
- **Architecture:** PhoBERT encoder + 2 independent linear heads (sentiment + aspect)
- **Training:** 10 epochs, AdamW, class weighting for imbalance

### Results on test set

| Task | Accuracy | Macro F1 |
|------|----------|----------|
| Sentiment | 0.578 | 0.51 |
| Aspect | 0.263 | 0.22 |

> Accuracy is affected by heavy class imbalance. Macro F1 reflects per-class performance better.

## Project Structure

```bash
viet-restaurant-absa/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 00_preparing_dataset.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── dashboard/
│   └── app.py
├── models/artifacts/
├── predict_demo.py
└── requirements.txt
```

## How to run

### Installation

```bash
git clone https://github.com/devin-ph/viet-restaurant-absa
cd viet-restaurant-absa
pip install -r requirements.txt
```

### Run demo:

```bash
python predict_demo.py "Quán này ngon lắm, phục vụ nhiệt tình"
```

### Train from scratch:
```bash
python src/train.py
```