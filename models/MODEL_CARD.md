# PhoBERT ABSA Restaurant

## Model Description

Fine-tuned [vinai/phobert-base](https://huggingface.co/vinai/phobert-base) for ABSA on Vietnamese restaurant reviews. The model predicts the aspect category and sentiment polarity from a given review.

## Dataset

- **Source:** VLSP2018-ABSA-Restaurant
- **Size:** Over 15k samples
- **Split:** 70% train / 10% val / 20% test

## Training

| Parameter | Value |
|-----------|-------|
| Base model | vinai/phobert-base |
| Max length | 128 |
| Batch size | 16 |
| Epochs | 10 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss with class weights |

## Results

| Task | Accuracy | Macro F1 |
|------|----------|----------|
| Sentiment | 0.578 | 0.51 |
| Aspect | 0.263 | 0.22 |

## Intended Use

- Analyzing Vietnamese restaurant reviews
- Identifying which aspect (food, service, price, ambience...) a review is about
- Determining sentiment toward that aspect

## Limitations

- Low accuracy on minority aspect classes (DRINKS#*, RESTAURANT#PRICES)
- Currently predicts one aspect per review
- Trained on restaurant domain only — may not generalize to other domains