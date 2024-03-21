import numpy as np
import random
from matplotlib import pyplot as plt
from modified_from_original_paper import *


# Set the training parameters
lda = 1.0
ep = 0.1
training_rounds = 1


def generate_random_structure(min_layers, max_layers, min_qubits, max_qubits):
    num_layers = random.randint(min_layers, max_layers)
    structure = [random.randint(min_qubits, max_qubits) for _ in range(num_layers)]
    return structure


def initialize_population(population_size, min_layers, max_layers, min_qubits, max_qubits):
    population = [generate_random_structure(min_layers, max_layers, min_qubits, max_qubits) for _ in
                  range(population_size)]
    return population

def fitness_function(individual, training_data):
    # Train the QNN with the given structure and evaluate its performance
    qnn_architecture = [2] + individual + [2]  # Add input and output layers
    initial_unitaries = [[]]  # Empty list for the input layer
    for i in range(1, len(qnn_architecture)):
        layer_unitaries = []
        for j in range(qnn_architecture[i]):
            unitary = randomQubitUnitary(qnn_architecture[i - 1] + 1)
            if qnn_architecture[i] - 1 != 0:
                unitary = qt.tensor(randomQubitUnitary(qnn_architecture[i - 1] + 1),
                                    tensoredId(qnn_architecture[i] - 1))
                unitary = swappedOp(unitary, qnn_architecture[i - 1], qnn_architecture[i - 1] + j)
            layer_unitaries.append(unitary)
        initial_unitaries.append(layer_unitaries)

    trained_unitaries = qnnTraining(qnn_architecture, initial_unitaries, training_data, lda, ep, training_rounds)
    output_states = feedforward(qnn_architecture, trained_unitaries[-1], training_data)
    cost = costFunction(training_data, output_states)
    return 1 / cost  # Minimize the cost, so higher fitness is better


def selection(population, fitness_scores):
    # Select individuals for reproduction based on their fitness scores
    selected_indices = random.choices(range(len(population)), weights=fitness_scores, k=len(population))
    selected_individuals = [population[i] for i in selected_indices]
    return selected_individuals


def crossover(parent1, parent2):
    # Perform crossover between two parent individuals to create offspring
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2


def mutation(individual, mutation_rate, min_qubits, max_qubits):
    # Perform mutation on an individual with a given mutation rate
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = random.randint(min_qubits, max_qubits)
    return mutated_individual


def genetic_algorithm(population_size, num_generations, mutation_rate, training_data, min_layers, max_layers,
                      min_qubits, max_qubits):
    population = initialize_population(population_size, min_layers, max_layers, min_qubits, max_qubits)
    best_fitness_values = []
    for generation, _ in enumerate(range(num_generations), start=1):
        fitness_scores = [fitness_function(individual, training_data) for individual in population]
        best_fitness = max(fitness_scores)
        best_fitness_values.append(best_fitness)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
        selected_individuals = selection(population, fitness_scores)
        new_population = [mutation(offspring, mutation_rate, min_qubits, max_qubits)
                          for parent1, parent2 in zip(selected_individuals[::2], selected_individuals[1::2])
                          for offspring in crossover(parent1, parent2)]
        population = new_population
    best_individual = max(population, key=lambda x: fitness_function(x, training_data))
    return best_individual, best_fitness_values