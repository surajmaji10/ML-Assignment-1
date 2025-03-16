import matplotlib.pyplot as plt

# Data points
items = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 
         55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]

random_strategy_times = [0.001505136489868164, 366.499342417, 742.963895581, 
                          1113.8556557789998, 1481.609303642, 1865.487979559, 
                          2285.019343032, 2712.460002185, 3141.869851798, 
                          3560.7853440930003, 3982.481387633, 4422.617822038999, 
                          4866.364187708999, 5292.054484208, 5717.834381369, 
                          6139.4216182499995, 6562.164641605, 6985.4221468999995, 
                          7409.334208446]

active_learning_times = [0.001505136489868164, 662.583816013, 1325.454193695, 
                         2013.423270873, 2767.980310034, 3514.657381953, 
                         4263.806902562, 5038.292986903, 5780.674877026, 
                         6521.193460914, 7269.14631303, 8000.290940605, 
                         8719.277065157, 9430.69309009, 10134.242320625, 
                         10800.212875079, 11459.455540989, 12135.219665172, 
                         12828.358868817]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(items, random_strategy_times, marker='o', linestyle='-', color='b', label="Random Strategy")
plt.plot(items, active_learning_times, marker='s', linestyle='-', color='r', label="Active Learning Strategy")

# Annotating values at each point
for i in range(len(items)):
    plt.text(items[i], random_strategy_times[i], f"{random_strategy_times[i]:.1f}", fontsize=8, verticalalignment='bottom', color='black')
    plt.text(items[i], active_learning_times[i], f"{active_learning_times[i]:.1f}", fontsize=8, verticalalignment='top', color='black')

# Labels and title
plt.xlabel("Number of Items")
plt.ylabel("Train Time (s)")
plt.title("Training Time Comparison: Random Strategy vs Active Learning")
plt.legend()
plt.grid(True)

plt.savefig('time_vs_items.png', dpi=300)
# Show the plot
# plt.show()
