import os
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from utils import *
from model import *


def main(args: argparse.Namespace):
    # Set seed for reproducibility
    set_seed(args.sr_no)

    # Load the data
    X_train, y_train, X_val, y_val = get_data(
        path=os.path.join(args.data_path, args.train_file), seed=args.sr_no)
    print("Data Loaded")

    # Preprocess the data
    vectorizer = Vectorizer(max_vocab_len=args.max_vocab_len)
    vectorizer.fit(X_train)

    if os.path.exists(f"{args.data_path}/X_train{args.intermediate}"):
        X_train_vec = pickle.load(open(f"{args.data_path}/X_train{args.intermediate}", "rb"))
        y_train = pickle.load(open(f"{args.data_path}/y_train{args.intermediate}", "rb"))
        X_val_vec = pickle.load(open(f"{args.data_path}/X_val{args.intermediate}", "rb"))
        y_val = pickle.load(open(f"{args.data_path}/y_val{args.intermediate}", "rb"))
        print("Preprocessed Data Loaded")
    else:
        X_train_vec = vectorizer.transform(X=X_train)
        pickle.dump(X_train_vec, open(f"{args.data_path}/X_train{args.intermediate}", "wb"))
        pickle.dump(y_train, open(f"{args.data_path}/y_train{args.intermediate}", "wb"))
        X_val_vec = vectorizer.transform(X=X_val)
        pickle.dump(X_val_vec, open(f"{args.data_path}/X_val{args.intermediate}", "wb"))
        pickle.dump(y_val, open(f"{args.data_path}/y_val{args.intermediate}", "wb"))
        print("Data Preprocessed")

    # Train the model
    start_train = time.time()
    model = MultinomialNaiveBayes(alpha=args.smoothing)
    model.fit(X_train_vec, y_train)
    end_train = time.time()
    train_time = end_train - start_train
    print(f"Model Trained in {train_time:.2f} seconds")

    # Evaluate the trained model
    start_val = time.time()
    y_pred_train = model.predict(X_train_vec)
    end_val = time.time()
    tt = end_val - start_val

    train_acc = np.mean(y_pred_train == y_train)

    start_val = time.time()
    y_pred_val = model.predict(X_val_vec)
    end_val = time.time()
    vt = end_val - start_val

    val_acc = np.mean(y_pred_val == y_val)
    end_val = time.time()
    

    print(f"Train Accuracy: {train_acc}")
    print(f"Validation Accuracy: {val_acc}")
    print(f"Training completed in {tt:.4f} seconds")
    print(f"Validation completed in {vt:.4f} seconds")

    # Load the test data
    if os.path.exists(f"{args.data_path}/X_test{args.intermediate}"):
        X_test_vec = pickle.load(open(f"{args.data_path}/X_test{args.intermediate}", "rb"))
        print("Preprocessed Test Data Loaded")
    else:
        X_test = pd.read_csv(f"{args.data_path}/X_test_{args.sr_no}.csv", header=None).values.squeeze()
        print("Test Data Loaded")
        X_test_vec = vectorizer.transform(X=X_test)
        pickle.dump(X_test_vec, open(f"{args.data_path}/X_test{args.intermediate}", "wb"))
        print("Test Data Preprocessed")

    # Predict on test data
    start_test = time.time()
    preds = model.predict(X_test_vec)
    end_test = time.time()
    test_time = end_test - start_test

    with open("predictions.csv", "w") as f:
        for pred in preds:
            f.write(f"{pred}\n")

    print(f"Predictions saved to predictions.csv")
    print(f"Test predictions completed in {test_time:.2f} seconds")
    print("You may upload the file at http://10.192.30.174:8000/submit")

    # Log times
    with open(f"{args.logs_path}/times.txt", "a") as f:
        f.write(f"Run {args.sr_no}: Train Time: {train_time:.2f}s, Validation Time: {val_time:.2f}s, Test Time: {test_time:.2f}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--intermediate", type=str, default="_i.pkl")
    parser.add_argument("--max_vocab_len", type=int, default=10_000)
    parser.add_argument("--smoothing", type=float, default=0.1)
    main(parser.parse_args())
