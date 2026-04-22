import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import torch
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LEN, NUM_SENTIMENTS, NUM_ASPECTS, MODEL_SAVE_PATH, SENTIMENT_MAP_PATH, ASPECT_MAP_PATH
from utils import get_path, load_pickle
from model import PhoBERTABSA
from predict import load_model, predict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_demo.py \"review text here\"")
        sys.exit(1)

    text = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sentiment_map = load_pickle(get_path(SENTIMENT_MAP_PATH))
    aspect_map = load_pickle(get_path(ASPECT_MAP_PATH))
    model = load_model(device)

    sentiment, aspect = predict(text, model, tokenizer, sentiment_map, aspect_map, device)
    print(f"Aspect   : {aspect}")
    print(f"Sentiment: {sentiment}")