import os

from utils import *
from model import *
import time

def main(args: argparse.Namespace):
    # set seed for reproducibility
    assert args.run_id is not None and 0 < args.run_id < 6, "Invalid run_id"
    set_seed(args.sr_no+args.run_id)

    # Load the preprocessed data
    if os.path.exists(f"{args.data_path}/X_train{args.intermediate}"):
        X_train_vec = pickle.load(open(
            f"{args.data_path}/X_train{args.intermediate}", "rb"))
        X_val_vec = pickle.load(open(
            f"{args.data_path}/X_val{args.intermediate}", "rb"))
        y_train = pickle.load(open(
            f"{args.data_path}/y_train{args.intermediate}", "rb"))
        y_val = pickle.load(open(
            f"{args.data_path}/y_val{args.intermediate}", "rb"))
        idxs =\
            np.random.RandomState(args.run_id).permutation(X_train_vec.shape[0])
        X_train_vec = X_train_vec[idxs]
        y_train = y_train[idxs]
        print("Preprocessed Data Loaded")
    else:
        raise Exception("Preprocessed Data not found")

    # Train the model
    model = MultinomialNaiveBayes(alpha=args.smoothing)
    accs = []
    total_items = 10_000
    idxs = np.arange(10_000)
    remaining_idxs = np.setdiff1d(np.arange(X_train_vec.shape[0]), idxs)

    start = time.process_time()

    # Train the model
    for i in range(1, 60):
        X_train_batch = X_train_vec[idxs]
        y_train_batch = y_train[idxs]

        if i == 1:
            model.fit(X_train_batch, y_train_batch)
        else:
            model.fit(X_train_batch, y_train_batch, update=True)

        end = time.process_time()
        taken_time = end - start
        with open(f"{args.logs_path}/run_{args.run_id}_{args.is_active}_time.txt", "a") as f:
            f.write(f"{total_items} items - Train time: {taken_time}\n")

        y_preds = model.predict(X_val_vec)
        val_acc = np.mean(y_preds == y_val)
        print(f"{total_items} items - Val acc: {val_acc}")
        accs.append(val_acc)

        y_preds = model.predict(X_train_vec)
        train_acc = np.mean(y_preds == y_train)
        print(f"{total_items} items - Train acc: {train_acc}")

        with open(f"{args.logs_path}/run_{args.run_id}_{args.is_active}_val.txt", "a") as f:
            f.write(f"{val_acc}\n")

        with open(f"{args.logs_path}/run_{args.run_id}_{args.is_active}_train.txt", "a") as f:
            f.write(f"{train_acc}\n")
        

        if args.is_active:
            # Active Learning: Select the most uncertain samples  
            X_unlabeled = X_train_vec[remaining_idxs]  
            log_probs = model.predict_(X_unlabeled)
            probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))  # Normalize  
            probs /= probs.sum(axis=1, keepdims=True)  # Ensure probabilities sum to 1   
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)  # Calculate entropy  
            uncertain_idxs = np.argsort(entropy)[-5_000:]  # Select top 5000 uncertain samples  
            idxs = np.concatenate([idxs, remaining_idxs[uncertain_idxs]])  
            remaining_idxs = np.delete(remaining_idxs, uncertain_idxs)
            # raise NotImplementedError
        else:
            # Random Sampling: Add the next 5000 samples
            idxs = np.concatenate([idxs, remaining_idxs[:5_000]])
            remaining_idxs = remaining_idxs[5_000:]

        total_items += 5_000
        
    accs = np.array(accs)
    np.save(f"{args.logs_path}/run_{args.run_id}_{args.is_active}.npy", accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--is_active", action="store_true")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--intermediate", type=str, default="_i.pkl")
    parser.add_argument("--max_vocab_len", type=int, default=10_000)
    parser.add_argument("--smoothing", type=float, default=0.1)
    main(parser.parse_args())
