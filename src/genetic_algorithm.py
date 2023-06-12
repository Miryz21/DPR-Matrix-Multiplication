import random
from math import floor
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle


m = 2
n = 2
d = 2
MAX_STEPS = m*n*d - 1
all_elems = m*n*n*d*m*d
# константы генетического алгоритма
POPULATION_SIZE = 200  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.01  # вероятность мутации индивидуума
MAX_GENERATIONS = 50  # максимальное количество поколений

# Импорт датасета с тензорами
with open('./tensor_holder/tensors', 'rb') as f:
    multiply_tensors = pickle.load(f)

T = multiply_tensors[(m, n, d)]

class FitnessMax():
    def __init__(self):
        self.values = [0]


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()


def findFitness(u, v, w):
    F = torch.clone(T)
    for i in range(MAX_STEPS):
        summ = torch.outer(torch.tensor(u[i])[0], torch.tensor(v[i])[0])
        for j in range(m*d):
            F[j] -= summ * np.array(w)[i][j]
    zeros = 0
    for x in F:
        for y in x:
            for z in y:
                zeros += 1 if int(z) == 0 else 0
    return zeros,


def individualCreator(a, b, c, n):
    n_u = np.matrix(''.join((''.join((('-1 ', '0 ', '1 ')[random.randrange(3)] for i in range(a * b))).strip() + ";")
                            for j in range(n))[:-1])
    n_v = np.matrix(''.join((''.join((('-1 ', '0 ', '1 ')[random.randrange(3)] for i in range(b * c))).strip() + ";")
                            for j in range(n))[:-1])
    n_w = np.matrix(''.join((''.join((('-1 ', '0 ', '1 ')[random.randrange(3)] for i in range(a * c))).strip() + ";")
                            for j in range(n))[:-1])

    return Individual([n_u, n_v, n_w])


def populationCreator(a, b, c, n, ln):
    return list([individualCreator(a, b, c, n) for i in range(ln)])


population = populationCreator(m, n, d, MAX_STEPS, POPULATION_SIZE)
fitnessVals = list(map(lambda el: findFitness(*el), [(population[i][0], population[i][1], population[i][2]) for i in range(POPULATION_SIZE)]))

for individual, fitnessVal in zip(population, fitnessVals):
    individual.fitness.values = fitnessVal


def clone(m):
    ind = Individual(m[:])
    ind.fitness.values[0] = m.fitness.values[0]
    return ind


def selTournament(populat, p_len):
    offspring = []
    # for n in range(p_len):
    #     i1 = i2 = i3 = 0
    #     while i1 == i2 or i1 == i3 or i2 == i3:
    #         i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)
    #     offspring.append(max([populat[i1], populat[i2], populat[i3]], key=lambda ind: ind.fitness.values[0]))
    offspring = sorted(populat, key=lambda ind: ind.fitness.values[0], reverse=True)[:p_len]

    return offspring


def cxOnePoint(child1, child2):
    lengths = [m*n, n*d, m*d]
    for c in range(random.randrange(3)):
        for i in range(MAX_STEPS):
            if random.randrange(10) >= 6:
                half = floor(lengths[c]/2)
                child1[c][i, :half], child2[c][i, lengths[c]-half:] = child2[c][i, lengths[c]-half:], child1[c][i, :half]
        # child1[c][l, r:], child2[c][l, r:] = child2[c][l, r:], child1[c][l, r:]


def mutFlipBit(mutant, indpb=0.01):
    lengths = [m*n, n*d, m*d]
    for c in range(3):
        indl = random.randint(0, MAX_STEPS - 1)
        indr = random.randint(0, lengths[c] - 1)
        if random.random() < indpb:
            numbers = [-1, 0, 1]
            mutant[c][indl, indr] = random.choice(numbers)


maxFitnessVals = []
meanFitnessVals = []
genCount = 0
maxFitness = 0
maxFitnessPrev = 0
fitnessVals = [individual.fitness.values[0] for individual in population]
maxCount = 0
while max(fitnessVals) < all_elems and genCount < 100000:
    # if genCount % 1000 == 0:
        # print(genCount)
    genCount += 1
    offspring = selTournament(population, POPULATION_SIZE)
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            # print("before:")
            # print(child1)
            # print(child2)
            cxOnePoint(child1, child2)
            # print("after:")
            # print(child1)
            # print(child2)

    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutFlipBit(mutant, indpb=1.0/all_elems)

    freshFitnessValues = list(map(lambda el: findFitness(*el), [(population[i][0], population[i][1], population[i][2]) for i in range(POPULATION_SIZE)]))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    population[:] = offspring

    fitnessVals = [ind.fitness.values[0] for ind in population]

    maxFitness = max(fitnessVals)
    if maxFitness == maxFitnessPrev:
        maxCount += 1
    else:
        maxCount = 0
    # if maxCount >= 50:
    #     for i in range(floor(0.2 * POPULATION_SIZE)):
    #         population[random.randrange(POPULATION_SIZE)] = individualCreator(2, 2, MAX_STEPS)
    maxFitnessPrev = maxFitness
    meanFitness = sum(fitnessVals) / len(population)
    maxFitnessVals.append(maxFitness)
    meanFitnessVals.append(meanFitness)
    print(f"Поколение {genCount}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}")

    best_index = fitnessVals.index(max(fitnessVals))
    print("Лучший индивидуум = ", *population[best_index], "\n")
    # l1.set_ydata(maxFitnessVals)
    # l2.set_ydata(meanFitness)
    # fig.canvas.draw()
    # fig.canvas.flush_events()


print(f"Поколение {genCount}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}")

print("Лучший индивидуум = ", *population[best_index], "\n")
