import os  
import argparse  
import numpy as np  
import matplotlib.pyplot as plt  

def main(args: argparse.Namespace):  
    assert os.path.exists(args.logs_path), "Invalid logs path"  

    # Load data for random and active strategies  
    random_means = []  
    random_variances = []  
    active_means = []  
    active_variances = []  

    for i in range(1, 6):  
        # Load random strategy data  
        random_file = os.path.join(args.logs_path, f"run_{i}_False.npy")  
        assert os.path.exists(random_file), f"File {random_file} not found"  
        random_data = np.load(random_file)
        print("RANDOM:\n", random_data)  
        random_means.append(np.round(np.mean(random_data), 4))  
        random_variances.append(np.round(np.var(random_data), 4))  

        # Load active strategy data  
        active_file = os.path.join(args.logs_path, f"run_{i}_True.npy")  
        assert os.path.exists(active_file), f"File {active_file} not found"  
        active_data = np.load(active_file) 
        print("ACTIVE:\n", active_data)   
        active_means.append(np.round(np.mean(active_data), 4))  
        active_variances.append(np.round(np.var(active_data), 4))  

    # X-axis labels for runs  
    runs = [f"Run {i}" for i in range(1, 6)]  
    x = np.arange(len(runs))  # the label locations  
    width = 0.35  # the width of the bars  

    # Plot for Means  
    plt.figure(figsize=(12, 6))  
    bars1 = plt.bar(x - width / 2, random_means, width, label="Random Strategy Mean", color="yellow")  
    bars2 = plt.bar(x + width / 2, active_means, width, label="Active Learning Strategy Mean", color="green")  

    # Add a horizontal dotted line for supervised accuracy  
    plt.axhline(y=args.supervised_accuracy/100, color='black', linestyle='dashed', linewidth=2, label="Supervised Accuracy (0.7603)")  

    # Annotate bars with mean values  
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() / 2,  # Position at the center of the bar
                f"{bar.get_height():.4f}", 
                ha='center', va='center',  # Align center horizontally and vertically
                fontsize=10, fontweight='bold', color='black')  # Use white for better visibility

    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() / 2,  
                f"{bar.get_height():.4f}", 
                ha='center', va='center',  
                fontsize=10, fontweight='bold', color='black')  


    # Labels, title, and legend  
    plt.xlabel("Runs")  
    plt.ylabel("Mean Accuracy")  
    plt.title("Mean Accuracy Across 5 Runs for Random and Active Learning Strategies")  
    plt.xticks(x, runs)  
    plt.legend()  
    plt.grid(True)  

    # Save the plot  
    means_plot_path = os.path.join(args.logs_path, f"means_plot_{args.sr_no}.png")  
    plt.savefig(means_plot_path, dpi=300)  
    print(f"Means plot saved to {means_plot_path}")  
    plt.close()  

    # Plot for Variances  
    plt.figure(figsize=(12, 6))  
    bars1 = plt.bar(x - width / 2, random_variances, width, label="Random Strategy Variance", color="yellow")  
    bars2 = plt.bar(x + width / 2, active_variances, width, label="Active Learning Strategy Variance", color="green")  

    # Annotate bars with variance values  
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.4f}", 
                 ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.4f}", 
                 ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    # Labels, title, and legend  
    plt.xlabel("Runs")  
    plt.ylabel("Variance")  
    plt.title("Variance Across 5 Runs for Random and Active Learning Strategies")  
    plt.xticks(x, runs)  
    plt.legend()  
    plt.grid(True)  

    # Save the plot  
    variances_plot_path = os.path.join(args.logs_path, f"variances_plot_{args.sr_no}.png")  
    plt.savefig(variances_plot_path, dpi=300)  
    print(f"Variances plot saved to {variances_plot_path}")  
    plt.close()  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--sr_no", type=int, required=True)  
    parser.add_argument("--logs_path", type=str, default="logs")  
    parser.add_argument("--supervised_accuracy", type=float, required=True)  
    main(parser.parse_args())  
