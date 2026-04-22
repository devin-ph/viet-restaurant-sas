import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LEN, NUM_SENTIMENTS, NUM_ASPECTS, MODEL_SAVE_PATH, SENTIMENT_MAP_PATH, ASPECT_MAP_PATH
from utils import get_path, load_pickle
from model import PhoBERTABSA


def load_model(device):
    model = PhoBERTABSA(MODEL_NAME, NUM_SENTIMENTS, NUM_ASPECTS)
    model.load_state_dict(torch.load(get_path(MODEL_SAVE_PATH), map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model


def predict(text, model, tokenizer, sentiment_map, aspect_map, device):
    inputs = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        sentiment_logits, aspect_logits = model(input_ids, attention_mask)

    sentiment_idx = sentiment_logits.argmax(dim=1).item()
    aspect_idx = aspect_logits.argmax(dim=1).item()

    sentiment_label = {v: k for k, v in sentiment_map.items()}[sentiment_idx]
    aspect_label = {v: k for k, v in aspect_map.items()}[aspect_idx]

    return sentiment_label, aspect_label


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sentiment_map = load_pickle(get_path(SENTIMENT_MAP_PATH))
    aspect_map = load_pickle(get_path(ASPECT_MAP_PATH))
    model = load_model(device)

    # test
    test_texts = [
        "Quán này ngon lắm, đồ ăn tuyệt vời",
        "Phục vụ tệ, nhân viên thái độ xấu",
        "Giá cả bình thường, không có gì đặc biệt"
    ]

    for text in test_texts:
        sentiment, aspect = predict(text, model, tokenizer, sentiment_map, aspect_map, device)
        print(f"Text: {text[:60]}")
        print(f"  → Aspect: {aspect}, Sentiment: {sentiment}\n")