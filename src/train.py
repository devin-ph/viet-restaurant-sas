import sys
import os
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from sklearn.utils.class_weight import compute_class_weight
    from tqdm import tqdm

    from config import (MODEL_NAME, BATCH_SIZE, EPOCHS,
                        LEARNING_RATE, NUM_SENTIMENTS, NUM_ASPECTS,
                        TRAIN_PATH, VAL_PATH, MODEL_SAVE_PATH)
    from utils import get_path
    from dataset import get_dataloader
    from model import PhoBERTABSA

    train_csv = os.path.normpath(get_path(TRAIN_PATH))
    val_csv = os.path.normpath(get_path(VAL_PATH))
    train_pt = os.path.normpath(get_path("data", "processed", "train_enc.pt"))
    val_pt = os.path.normpath(get_path("data", "processed", "val_enc.pt"))

    print("Loading CSVs...")
    train_df = pd.read_csv(train_csv)
    print("  train_df:", train_df.shape)
    val_df = pd.read_csv(val_csv)
    print("  val_df:", val_df.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    print("Loading encodings...")
    train_enc = torch.load(train_pt, weights_only=False)
    val_enc = torch.load(val_pt, weights_only=False)
    print("  encodings ok")

    train_loader = get_dataloader(train_enc, train_df['sentiment'].values,
                                  train_df['aspect'].values, BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_enc, val_df['sentiment'].values,
                                val_df['aspect'].values, BATCH_SIZE)

    sentiment_classes = np.array(range(NUM_SENTIMENTS))
    sentiment_weights = compute_class_weight('balanced',
                                             classes=sentiment_classes,
                                             y=train_df['sentiment'].values)
    sentiment_weights = torch.tensor(sentiment_weights, dtype=torch.float).to(device)

    aspect_classes = np.array(range(NUM_ASPECTS))
    aspect_weights = compute_class_weight('balanced',
                                          classes=aspect_classes,
                                          y=train_df['aspect'].values)
    aspect_weights = torch.tensor(aspect_weights, dtype=torch.float).to(device)

    print("Class weights computed")

    print("Loading model...")
    model = PhoBERTABSA(MODEL_NAME, NUM_SENTIMENTS, NUM_ASPECTS).to(device)
    print("  model ok")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_weights)
    aspect_criterion = nn.CrossEntropyLoss(weight=aspect_weights)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    save_path = os.path.normpath(get_path(MODEL_SAVE_PATH))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # train
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc="  train", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            aspect_labels = batch['aspect_labels'].to(device)

            optimizer.zero_grad()
            sentiment_logits, aspect_logits = model(input_ids, attention_mask)

            loss = sentiment_criterion(sentiment_logits, sentiment_labels) + aspect_criterion(aspect_logits, aspect_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        # evaluate
        model.eval()
        total_val_loss = 0
        correct_sentiment = 0
        correct_aspect = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  eval", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)
                aspect_labels = batch['aspect_labels'].to(device)

                sentiment_logits, aspect_logits = model(input_ids, attention_mask)

                total_val_loss += (sentiment_criterion(sentiment_logits, sentiment_labels) + aspect_criterion(aspect_logits, aspect_labels)).item()
                correct_sentiment += (sentiment_logits.argmax(dim=1) == sentiment_labels).sum().item()
                correct_aspect += (aspect_logits.argmax(dim=1) == aspect_labels).sum().item()
                total += len(sentiment_labels)

        val_loss = total_val_loss / len(val_loader)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Sentiment Acc: {correct_sentiment / total:.4f}")
        print(f"  Val Aspect Acc: {correct_aspect / total:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("  Model saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()