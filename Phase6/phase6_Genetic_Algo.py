import numpy as np
import matplotlib.pyplot as plt
import random

N = 8
POP_SIZE = 200
GENERATIONS = 1000
MUTATION_RATE = 0.3

def fitness(board):
    attacks = 0
    for i in range(N):
        for j in range(i+1, N):
            if board[i] == board[j] or abs(board[i]-board[j]) == j-i:
                attacks += 1
    return 28 - attacks

def create_board():
    return [random.randint(0,7) for _ in range(8)]

def crossover(p1, p2):
    point = random.randint(1,7)
    return p1[:point] + p2[point:]

def mutate(board):
    if random.random() < MUTATION_RATE:
        board[random.randint(0,7)] = random.randint(0,7)
    return board

population = [create_board() for _ in range(POP_SIZE)]
best_fitness = []

solution = None

for gen in range(GENERATIONS):
    population.sort(key=lambda x: fitness(x), reverse=True)
    best = population[0]
    best_fitness.append(fitness(best))

    if fitness(best) == 28:
        solution = best
        print("🎉 Solution Found:", best)
        break

    new_pop = population[:50]
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(population[:100], 2)
        child = mutate(crossover(p1,p2))
        new_pop.append(child)

    population = new_pop

if solution is None:
    solution = population[0]
    print("Best Approximate Solution:", solution)

plt.plot(best_fitness)
plt.title("Phase-6 Genetic Algorithm Convergence")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.savefig("phase6_convergence.png")
plt.show()

print("\nphase6_convergence.png saved successfully.")
