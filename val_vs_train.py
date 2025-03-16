import matplotlib.pyplot as plt

# Data
items = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 
         55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]

# Random Strategy Accuracy
random_val_acc = [0.7056, 0.7140, 0.7231, 0.7279, 0.7305, 0.7340, 0.7361, 0.7387, 
                  0.7403, 0.7415, 0.7431, 0.7443, 0.7450, 0.7458, 0.7464, 0.7473, 
                  0.7475, 0.7482, 0.7488]

random_train_acc = [0.7078, 0.7154, 0.7241, 0.7291, 0.7324, 0.7355, 0.7374, 0.7397, 
                    0.7416, 0.7431, 0.7445, 0.7458, 0.7468, 0.7477, 0.7483, 0.7488, 
                    0.7492, 0.7496, 0.7502]

# Active Learning Strategy Accuracy
active_val_acc = [0.7056, 0.7132, 0.7191, 0.7222, 0.7257, 0.7284, 0.7303, 0.7328, 
                  0.7351, 0.7363, 0.7376, 0.7387, 0.7395, 0.7411, 0.7422, 0.7433, 
                  0.7439, 0.7444, 0.7450]

active_train_acc = [0.7078, 0.7153, 0.7207, 0.7249, 0.7286, 0.7318, 0.7346, 0.7373, 
                    0.7391, 0.7409, 0.7427, 0.7444, 0.7457, 0.7469, 0.7481, 0.7495, 
                    0.7504, 0.7515, 0.7525]

# Plot
plt.figure(figsize=(12, 6))

# Random Strategy
plt.plot(items, random_val_acc, 'bo-', label="Random Validation Accuracy")  
plt.plot(items, random_train_acc, 'b--', label="Random Training Accuracy")  

# Active Learning Strategy
plt.plot(items, active_val_acc, 'ro-', label="Active Learning Validation Accuracy")  
plt.plot(items, active_train_acc, 'r--', label="Active Learning Training Accuracy")  

# Labels and Title
plt.xlabel("Number of Training Items", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Validation & Training Accuracy for Random vs Active Learning", fontsize=14)
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("accuracy_comparison.png", dpi=300)
print("Plot saved as 'accuracy_comparison.png'")

# Show the plot
# plt.show()
