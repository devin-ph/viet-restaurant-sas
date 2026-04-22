import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
import gradio as gr
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LEN, NUM_SENTIMENTS, NUM_ASPECTS, MODEL_SAVE_PATH, SENTIMENT_MAP_PATH, ASPECT_MAP_PATH
from utils import get_path, load_pickle
from model import PhoBERTABSA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sentiment_map = load_pickle(get_path(SENTIMENT_MAP_PATH))
aspect_map = load_pickle(get_path(ASPECT_MAP_PATH))

inv_sentiment = {v: k for k, v in sentiment_map.items()}
inv_aspect = {v: k for k, v in aspect_map.items()}

model = PhoBERTABSA(MODEL_NAME, NUM_SENTIMENTS, NUM_ASPECTS)
model.load_state_dict(torch.load(get_path(MODEL_SAVE_PATH), map_location=device, weights_only=False))
model.to(device)
model.eval()


def predict(text):
    if not text.strip():
        return "Please enter a review.", ""

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

    sentiment = inv_sentiment[sentiment_idx]
    aspect = inv_aspect[aspect_idx]

    sentiment_emoji = {"positive": "😊 Positive", "neutral": "😐 Neutral", "negative": "😞 Negative"}

    return sentiment_emoji.get(sentiment, sentiment), aspect


examples = [
    ["Đồ ăn ngon lắm, tôi rất thích món bò lúc lắc ở đây"],
    ["Phục vụ tệ, nhân viên thái độ rất xấu"],
    ["Giá cả hơi cao so với chất lượng"],
    ["Không gian quán đẹp, thoáng mát, yên tĩnh"],
    ["Nước uống ở đây bình thường thôi, không có gì đặc biệt"],
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter restaurant review here...",
        label="Review"
    ),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Aspect"),
    ],
    title="Vietnamese Restaurant Review ABSA",
    description="Analyzing sentiment and aspect of Vietnamese restaurant reviews using PhoBERT.",
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()