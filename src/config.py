import os

MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 128

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5

NUM_SENTIMENTS = 3
NUM_ASPECTS = 12

TRAIN_PATH = os.path.join("data", "processed", "train_encoded.csv")
VAL_PATH = os.path.join("data", "processed", "val_encoded.csv")
TEST_PATH = os.path.join("data", "processed", "test_encoded.csv")

MODEL_SAVE_PATH = os.path.join("models", "artifacts", "phobert_absa.pt")
SENTIMENT_MAP_PATH = os.path.join("models", "artifacts", "sentiment_map.pkl")
ASPECT_MAP_PATH = os.path.join("models", "artifacts", "aspect_map.pkl")