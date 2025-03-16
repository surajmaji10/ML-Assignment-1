import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args: argparse.Namespace):
    assert os.path.exists(args.logs_path), "Invalid logs path"
    # for i in [True, False]:
    #     for j in range(1, 6):
    #         assert os.path.exists(os.path.join(args.logs_path, f"run_{j}_{i}.npy")),\
    #             f"File run_{j}_{i}.npy not found in {args.logs_path}"
    # TODO: Load data and plot the standard means and standard deviations of
    # the accuracies for the two settings (active and random strategies)
    # TODO: also ensure that the files have the same length

    # Load data for random and active strategies  
    random_accs = []  
    active_accs = []  

    for i in range(1, 6):  
        # Load random strategy data  
        random_file = os.path.join(args.logs_path, f"run_{1}_False.npy")  
        assert os.path.exists(random_file), f"File {random_file} not found"  
        random_accs.append(np.load(random_file))  

        # Load active strategy data  
        active_file = os.path.join(args.logs_path, f"run_{1}_False.npy")  
        assert os.path.exists(active_file), f"File {active_file} not found"  
        active_accs.append(np.load(active_file))  

    # Ensure all files have the same length  
    min_length = min(min(len(acc) for acc in random_accs), min(len(acc) for acc in active_accs))  
    random_accs = [acc[:min_length] for acc in random_accs]  
    active_accs = [acc[:min_length] for acc in active_accs]  

    # Compute mean and standard deviation  
    random_mean = np.mean(random_accs, axis=0)  
    random_std = np.std(random_accs, axis=0)  
    active_mean = np.mean(active_accs, axis=0)  
    active_std = np.std(active_accs, axis=0)  

    print(random_accs)
    print(active_accs)
    print(random_mean)
    print(random_std)
    print(active_mean)
    print(active_std)

    # Plot the results  
    x = np.arange(10_000, 10_000 + min_length * 5_000, 5_000)  # X-axis: number of labeled samples  
    plt.figure(figsize=(10, 6))  

    # Plot random strategy  
    plt.plot(x, random_mean, label="Random Strategy", color="blue")  
    plt.fill_between(x, random_mean - random_std, random_mean + random_std, color="blue", alpha=0.2)  

    # Plot active strategy  
    plt.plot(x, active_mean, label="Active Learning Strategy", color="red")  
    plt.fill_between(x, active_mean - active_std, active_mean + active_std, color="red", alpha=0.2)  

    # Plot supervised accuracy  
    plt.axhline(y=args.supervised_accuracy, color="green", linestyle="dotted", label="Supervised Accuracy")  

    # Add labels and title  
    plt.xlabel("Number of Labeled Samples")  
    plt.ylabel("Validation Accuracy")  
    plt.title("Comparison of Random vs Active Learning Strategy")  
    plt.legend()  
    plt.grid(True)  

    # Save the plot  
    plot_path = os.path.join(args.logs_path, f"comparison_plot_{args.sr_no}.png")  
    plt.savefig(plot_path)  
    print(f"Plot saved to {plot_path}") 
    return
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--supervised_accuracy", type=float, required=True)
    main(parser.parse_args())
