import csv
import math
import random
from dataclasses import dataclass, field
import time
import sys

import matplotlib.pyplot as plt

from typing import List

NO_GENERATIONS = 450
POPULATION_SIZE = 45
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.35
NO_OF_MUTATIONS = 7
OVER_WEIGHT_PENALTY = 1000
KEEP_BEST = True

NO_VEHICLES = 9
MAX_VEHICLE_WEIGHT = 100
NO_EXPERIMENT_ITERATIONS = 20


@dataclass
class Chromosome:
    stops: List[int] = field(default_factory=list)
    vehicles: List[int] = field(default_factory=list)
    fitness: int = 0


# generates a single chromosome
def gen_chromosome():
    # generates a random sequence of customers and a corresponding array which vehicle stops at this customer
    _stops = list(range(1, 55))
    random.shuffle(_stops)
    _vehicles = []
    for i in range(0, len(_stops)):
        _vehicles.append(random.randint(0, 8))

    return Chromosome(_stops.copy(), _vehicles.copy())


# generates the initial population
def gen_population():
    _chromosomes = []
    for i in range(0, POPULATION_SIZE):
        _chromosomes.append(gen_chromosome())
    return _chromosomes


# calculates the weights of the vehicles in one chromosome
def calculate_weights(c: Chromosome):
    vehicle_weights = [0] * NO_VEHICLES
    for i in range(0, len(c.vehicles)):
        stop = c.stops[i]  # the current stop
        vehicle_no = c.vehicles[i]  # the current driver that stops for this customer
        vehicle_weights[vehicle_no] += demands[stop]
    return vehicle_weights


# calculate the path_costs and vehicle_weights
def calculate_path_costs_and_weights(c: Chromosome):
    path_costs = [0] * NO_VEHICLES
    vehicle_weights = [0] * NO_VEHICLES
    prev_stop = [0] * NO_VEHICLES

    for i in range(0, len(c.vehicles)):
        stop = c.stops[i]  # the current stop
        vehicle_no = c.vehicles[i]  # the current driver that stops for this customer
        dist = distance_matrix[prev_stop[vehicle_no]][stop]  # distance driver makes for this customer
        path_costs[vehicle_no] += dist
        vehicle_weights[vehicle_no] += demands[stop]
        prev_stop[vehicle_no] = stop

    # calculate costs for return to depot
    for i in range(0, len(prev_stop)):
        return_dist = distance_matrix[prev_stop[i]][0]
        path_costs[i] += return_dist

    return path_costs, vehicle_weights


# applying the fitness function using a penalty if the weight of a vehicle is more than 100
def fun_fitness(costs, weight):
    if weight > 100:
        weight_penalty = (weight - 100) * OVER_WEIGHT_PENALTY
    else:
        weight_penalty = 0
    fitness = costs + weight_penalty
    return fitness


# calculates the fitness of a chromosome, the higher the fitness the better is the chromosome
def evaluate_fitness(c: Chromosome):
    path_costs, vehicle_weights = calculate_path_costs_and_weights(c)

    total_fitness = 0
    for i in range(0, len(path_costs)):
        f = fun_fitness(path_costs[i], vehicle_weights[i])
        total_fitness += f

    c.fitness = 1 / total_fitness


# select the parent using the roulette wheel selection
def select_parent(chromosomes):
    total_fitness = 0
    chroms_fitness = []
    for chrom in chromosomes:
        total_fitness += chrom.fitness
        chroms_fitness.append(chrom.fitness)

    # create the selection probabilities from the scaled fitness
    selection_probabilities = [f_s / total_fitness for f_s in chroms_fitness]

    selected_chrom = random.choices(chromosomes, weights=selection_probabilities)[0]

    return selected_chrom


# do the crossover, implemented according to the order crossover
def do_crossover(parent1: Chromosome, parent2: Chromosome):
    crossover_point_1 = random.randint(0, len(parent1.stops) - 1)
    crossover_point_2 = random.randint(0, len(parent1.stops) - 1)
    child_stops = [-1] * len(parent1.stops)
    used_values = []

    for i in range(min(crossover_point_1, crossover_point_2), max(crossover_point_1, crossover_point_2) + 1):
        child_stops[i] = parent1.stops[i]
        used_values.append(parent1.stops[i])

    available_values = [ele for ele in parent2.stops if ele not in used_values]

    for i in range(0, len(parent1.stops)):
        if child_stops[i] == -1:
            child_stops[i] = available_values.pop(0)

    child_1 = Chromosome(child_stops, parent1.vehicles.copy())
    child_2 = Chromosome(child_stops, parent2.vehicles.copy())

    evaluate_fitness(child_1)
    evaluate_fitness(child_2)

    if child_1.fitness > child_2.fitness:
        return child_1
    else:
        return child_2


# does the mutation by swapping to random elements
def do_mutation(c: Chromosome):

    old_chrom = Chromosome(c.stops.copy(), c.vehicles.copy(), c.fitness)

    for j in range(0, NO_OF_MUTATIONS):
        if random.uniform(0, 1) < MUTATION_RATE:
            rand = random.uniform(0, 1)
            if rand < 0.2:
                swap_gene_stops(c)
            elif rand < 0.4:
                swap_gene_vehicles(c)
            elif rand < 0.6:
                exchange_gene_vehicles(c)
            elif rand < 0.8:
                two_opt_one_path(c)
            else:
                two_opt_two_paths(c)

        evaluate_fitness(c)
        if c.fitness < old_chrom.fitness:
            c.stops = old_chrom.stops.copy()
            c.vehicles = old_chrom.vehicles.copy()
        else:
            old_chrom.stops = c.stops.copy()
            old_chrom.vehicles = c.vehicles.copy()
            old_chrom.fitness = c.fitness


# swaps two genes in the stops array
def swap_gene_stops(c: Chromosome):
    swapping_index_1 = random.randint(0, len(c.stops) - 1)
    swapping_index_2 = random.randint(0, len(c.stops) - 1)

    temp_stop = c.stops[swapping_index_1]
    c.stops[swapping_index_1] = c.stops[swapping_index_2]
    c.stops[swapping_index_2] = temp_stop


# swaps two genes in the vehicles array
def swap_gene_vehicles(c: Chromosome):
    swapping_index_1 = random.randint(0, len(c.stops) - 1)
    swapping_index_2 = random.randint(0, len(c.stops) - 1)

    temp_vehicle = c.vehicles[swapping_index_1]
    c.vehicles[swapping_index_1] = c.vehicles[swapping_index_2]
    c.vehicles[swapping_index_2] = temp_vehicle


# exchanges the allele of one gene in the vehicles with a random other one if it is feasible
def exchange_gene_vehicles(c: Chromosome):
    changing_index = random.randint(0, len(c.stops) - 1)
    vehicle_weights = calculate_weights(c)
    demand_at_changing_index = demands[changing_index]
    available_vehicles = []

    for i in range(0, len(vehicle_weights)):
        if demand_at_changing_index + vehicle_weights[i] < MAX_VEHICLE_WEIGHT:
            available_vehicles.append(i)

    if len(available_vehicles) > 0:
        c.vehicles[changing_index] = random.choice(available_vehicles)


# calculates the costs if the route is changed with these four stops
def cost_route_change(stop1, stop2, stop3, stop4):
    return distance_matrix[stop1][stop3] + distance_matrix[stop2][stop4] - distance_matrix[stop1][stop2] - distance_matrix[stop3][stop4]


# implements part of the 2-opt route algorithm to avoid twists in a route
def two_opt_route(route):
    path_size = len(route)
    for i in range(1, path_size - 2):
        for j in range(i + 1, path_size):
            if j - i == 1:
                continue
            if cost_route_change(route[i - 1], route[i], route[j - 1], route[j]) < 0:
                route[i:j] = route[j - 1:i - 1:-1]
    return route


# removes twists in the path of one vehicle in a chromosome
def two_opt_one_path(c: Chromosome):
    vehicle_to_check = random.randint(0, 8)
    route = [0]
    for i in range(0, len(c.vehicles)):
        if c.vehicles[i] == vehicle_to_check:
            route.append(c.stops[i])

    route.append(0)
    route = two_opt_route(route)
    route.pop(0)

    for i in range(0, len(c.vehicles)):
        if c.vehicles[i] == vehicle_to_check:
            c.stops[i] = route.pop(0)


# redistributes the stops between two paths and reorders them using 2-opt
def two_opt_two_paths(c: Chromosome):
    vehicle_1 = random.randint(0, 8)
    vehicle_2 = random.randint(0, 8)
    if vehicle_1 == vehicle_2:
        return

    # return if overweight to keep only feasible solutions
    vehicle_weights = calculate_weights(c)
    if vehicle_weights[vehicle_1] + vehicle_weights[vehicle_2] >= 2 * MAX_VEHICLE_WEIGHT:
        return

    # get all the stops of the two vehicles
    stops = []
    for i in range(0, len(c.vehicles)):
        if c.vehicles[i] == vehicle_1 or c.vehicles[i] == vehicle_2:
            stops.append(c.stops[i])

    # calculate the max dist of all stops in the routes to select them as starting stops for the individual routes
    max_dist = 0
    max_point1 = -1
    max_point2 = -1
    for i in range(0, len(stops) - 1):
        for j in range(i+1, len(stops)):
            dist = distance_matrix[stops[i]][stops[j]]
            if dist > max_dist:
                max_dist = dist
                max_point1 = stops[i]
                max_point2 = stops[j]

    # initialize new routes and weights with zero
    route1 = [0]
    route2 = [0]
    weights = [0, 0]
    route1.append(max_point1)
    route2.append(max_point2)
    weights[0] += demands[max_point1]
    weights[1] += demands[max_point2]
    # remove the max points from the stops array
    stops.remove(max_point1)
    stops.remove(max_point2)

    # add the next stop to the closest route if possible
    for i in range(0, len(stops)):
        dist1 = distance_matrix[max_point1][stops[i]]
        dist2 = distance_matrix[max_point2][stops[i]]

        # if the current stop is closer to route of v1 then we add it there as long as it is not full and vice versa
        if dist1 < dist2:
            if weights[0] + demands[stops[i]] <= MAX_VEHICLE_WEIGHT:
                weights[0] += demands[stops[i]]
                route1.append(stops[i])
            else:
                weights[1] += demands[stops[i]]
                route2.append(stops[i])
        else:
            if weights[1] + demands[stops[i]] <= MAX_VEHICLE_WEIGHT:
                weights[1] += demands[stops[i]]
                route2.append(stops[i])
            else:
                weights[0] += demands[stops[i]]
                route1.append(stops[i])

    # insert endpoint for 2-opt routes function
    route1.append(0)
    route2.append(0)

    # free the route from twists
    route1 = two_opt_route(route1)
    route2 = two_opt_route(route2)

    # remove end and starting points from optimized routes
    route1 = route1[1:-1]
    route2 = route2[1:-1]

    # change original chromosome
    # traverse the vehicles array of the chromosome and change the current stop and vehicle according to new routes
    for i in range(0, len(c.vehicles)):
        if c.vehicles[i] == vehicle_1 or c.vehicles[i] == vehicle_2:
            if len(route1) > len(route2):
                c.stops[i] = route1.pop(0)
                c.vehicles[i] = vehicle_1
            else:
                c.stops[i] = route2.pop(0)
                c.vehicles[i] = vehicle_2


# returns the best chromosome in a population
def get_best_chromosome(population):
    max_fitness = - sys.maxsize
    best_chrom = Chromosome([], [])
    for c in population:
        if c.fitness > max_fitness:
            max_fitness = c.fitness
            best_chrom = c
    return best_chrom


# shows the phenotype of a chromosome
def print_phenotype(c: Chromosome):
    path_costs, vehicle_weights = calculate_path_costs_and_weights(c)
    print("The total costs of the paths are:", "{:.2f}".format(sum(path_costs)))
    print("Vehicle weights:", vehicle_weights)

    for i in range(0, NO_VEHICLES):
        print("Route #", i + 1, ":", sep="", end=" ")
        for j in range(0, len(c.vehicles)):
            if c.vehicles[j] == i:
                print(c.stops[j], end=" ")
        print("")


# implements the Genetic Algorithm
def ga_solve():
    curr_population = gen_population()
    for chrom in curr_population:
        evaluate_fitness(chrom)

    for i in range(0, NO_GENERATIONS):
        new_population = []
        for j in range(0, POPULATION_SIZE):
            parent1 = select_parent(curr_population)

            if random.uniform(0, 1) < CROSSOVER_RATE:
                parent2 = select_parent(curr_population)
                child = do_crossover(parent1, parent2)
            else:
                child = parent1

            do_mutation(child)
            evaluate_fitness(child)
            new_population.append(child)

        if KEEP_BEST:
            best = get_best_chromosome(curr_population)
            new_population[0] = best

        curr_population = new_population

    return get_best_chromosome(curr_population)


# plot the routes of the vehicles of a chromosome as a map
def plot_map(c: Chromosome, data):
    x_data = [d[2] for d in data]
    y_data = [d[3] for d in data]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:grey"]

    routes = []

    for i in range(0, NO_VEHICLES):
        route = [0]
        for j in range(0, len(c.vehicles)):
            if c.vehicles[j] == i:
                route.append(c.stops[j])
        route.append(0)
        routes.append(route)

    for i in range(0, len(routes)):
        x_points = []
        y_points = []
        for j in routes[i]:
            x_points.append(x_data[j])
            y_points.append(y_data[j])
        plt.plot(x_points[1:-1], y_points[1:-1], label="Route" + str(i + 1), marker='o', color=colors[i])
        plt.plot(x_points[:2], y_points[:2], color=colors[i], linestyle="--")
        plt.plot(x_points[-2:], y_points[-2:], color=colors[i], linestyle="--")

    plt.plot(x_data[0], y_data[0], marker='o', color='black')

    plt.legend()
    plt.show()


# print costs and weights of a single iteration of the best chromosome
def print_cost_and_weight(costs, weights, iteration, runtime):
    print("Iteration: ", iteration, " runtime: ", "{:.4f}".format(runtime), ", costs: ", "{:.2f}".format(sum(costs)), ", weights: ", weights, sep="")
    return sum(costs)


# calculates the distance based on euclidean metric measurement
def calc_dist(x1, y1, x2, y2):
    return math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))


# creates the distance matrix and the demands array
def calculate_map_context():
    file = open("data.csv")
    csvreader = csv.reader(file)
    next(csvreader)  # skip header
    rows = []
    for row in csvreader:
        rows.append(list(map(int, row)))

    _dist_matrix = []
    _demands = [0] * len(rows)
    for i in range(0, len(rows)):
        row = [0] * len(rows)
        _dist_matrix.append(row)

    for i in range(0, len(rows)):
        _demands[i] = rows[i][1]
        for j in range(i, len(rows)):
            start_x = rows[i][2]
            start_y = rows[i][3]
            end_x = rows[j][2]
            end_y = rows[j][3]
            dist = calc_dist(start_x, start_y, end_x, end_y)
            _dist_matrix[i][j] = dist
            _dist_matrix[j][i] = dist
    return _dist_matrix, _demands, rows


if __name__ == '__main__':

    distance_matrix, demands, data_matrix = calculate_map_context()

    best_chrom_runtime = 0
    best_chrom_total_cost = 0
    best_chromosome = Chromosome([], [])
    total_cpu_time = 0
    optimal_solution_costs = 1073

    for i in range(0, NO_EXPERIMENT_ITERATIONS):
        start_time = time.time()
        chromosome = ga_solve()
        end_time = time.time()

        costs_i, weights_i = calculate_path_costs_and_weights(chromosome)
        print_cost_and_weight(costs_i, weights_i, i + 1, end_time - start_time)
        print_phenotype(chromosome)
        total_cpu_time += end_time - start_time

        if chromosome.fitness > best_chromosome.fitness:
            best_chromosome = chromosome
            best_chrom_total_cost = sum(costs_i)
            best_chrom_runtime = end_time - start_time

    print("\nBest result in detail\n")
    print("Total CPU Time: ", "{:.2f}".format(total_cpu_time), "s", sep="")
    print("Total number of runs:", NO_EXPERIMENT_ITERATIONS)
    print("Runtime of the algorithm for the best solution: ", "{:.2f}".format(best_chrom_runtime), "s", sep="")
    print("Absolute difference of optimal solution:", "{:.2f}".format(best_chrom_total_cost - optimal_solution_costs))
    print("Relative difference of optimal solution: ", "{:.2f}".format((100 / optimal_solution_costs * best_chrom_total_cost) - 100), "%", sep="")
    print("Weights and routes of best solution:\n")

    print_phenotype(best_chromosome)
    plot_map(best_chromosome, data_matrix)
