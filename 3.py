import math
import random

min_val = 1
max_val = 10
target = 100
max_population_size = 200
epsilon = 1e-7
init_pop_size = 2
max_inters = 10000

def get_z(x, y):
    return pow(x, 2) + pow(math.e, y/5) + 100*math.log(x, 2) - 1 / pow(x, y) - x

def get_fitness(curr):
    return 1 - abs(target-curr)/(abs(target) + abs(curr))

def rand():
    return random.uniform(min_val, max_val)

def get_init_population(size):
    pop = []
    for _ in range(size):
        pop.append([rand(), rand()])
    return pop

def reproduce(x1, y1, x2, y2):
    return [[x1, x2], [y1, y2], [x1, y2], [x2, y1]]

def do_crossover(population):
    new_population = []
    i = 0
    while i < len(population)-1:
        new_population += reproduce(*population[i], *population[i+1])
        i += 2
    return new_population

def do_mutation(population):
    for i in range(len(population)):
        should_mutate = random.choice([True, False])
        if should_mutate:
            index = random.choice([0, 1])
            population[i][index] = rand()
    return population

def sort_by_fitness(population):
    return sorted(population, key=lambda i: get_fitness(get_z(i[0], i[1])), reverse=True)

def do_selection(population):
    population = sort_by_fitness(population)
    return population[:min(len(population), max_population_size)]

def main():
    pop = get_init_population(init_pop_size)
    ans = pop[0]
    fitness = get_fitness(get_z(*ans))
    i = 0
    while (i < max_inters and abs(fitness-1) > epsilon):
        i += 1
        print("Iteration number: " + str(i))
        pop = do_crossover(pop)
        pop = do_mutation(pop)
        pop = do_selection(pop)
        new_fitness = get_fitness(get_z(*pop[0]))
        if new_fitness > fitness:
            ans = pop[0]
            fitness = new_fitness
    
    print("Final answer: " + str(ans))


if __name__ == '__main__':
    main()

