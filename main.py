from modified_from_original_paper import *
from first_ga import *
from time import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
globals()


# Set the parameters for the genetic algorithm
population_size = 20
num_generations = 30
mutation_rate = 0.8
min_layers = 2
max_layers = 10
min_qubits = 2
max_qubits = 8

# Generate random training data
num_training_pairs = 20
random_unitary = randomQubitUnitary(2)
training_data = randomTrainingData(random_unitary, num_training_pairs)


start_time = time()  # Start the timer

# Run the genetic algorithm
best_structure, best_fitness_values = genetic_algorithm(population_size, num_generations, mutation_rate, training_data,
                                                        min_layers, max_layers, min_qubits, max_qubits)

end_time = time()  # End the timer
execution_time = end_time - start_time

print("Best QNN structure found:", [2] + best_structure + [2])
print("Execution time:", execution_time, "seconds")

# Plot the convergence curve
plt.plot(range(num_generations), best_fitness_values)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Convergence Plot")
plt.show()