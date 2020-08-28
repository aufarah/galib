import numpy as np
import copy
import galib

class custom:
    def __init__(self):
        self.weight = np.matrix([np.random.uniform(-10,10,2)]).T
        self.bias = np.random.uniform(-10,10,1)
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def forward(self,x):
        return self.sigmoid(x * self.weight + self.bias)
    
class test_galib(galib.genetic):
    def __init__(self,initialPop,numGeneration,numChild,numErase,bigBest,classModel):
        super().__init__(initialPop,numGeneration,numChild,numErase,bigBest,classModel)
        
    def crossover(self,population):
        for child in range(self.numChild):
            x = np.random.randint(len(population))
            new_individu = copy.deepcopy(self.population[x])
            x = np.random.randint(len(population))
            power = np.random.uniform(-10,10)
            powera = np.random.uniform(-10,10)
            new_individu.weight = (powera*new_individu.weight + power*self.population[x].weight) / 2
            new_individu.bias = (powera*new_individu.bias + power*self.population[x].bias) / 2
            self.population = np.append(self.population,new_individu)

    def findfit(self,population):
        fitness = np.array([])
        input = np.matrix([[0.,0.],
                           [0.,1.],
                           [1.,0.],
                           [1.,1.]])
        output = np.matrix([[0,1,1,1]]).T
                        
        for individu in population:
            result = individu.forward(input)
            error = result - output
            cost = error.T * error
            cost = cost[0,0]
            fitness = np.append(fitness,cost)
        return fitness

if __name__ == "__main__":
    tester = test_galib(initialPop=80,numGeneration=60,numChild=20,numErase=20,bigBest=False,classModel=custom)
    tester.train()

    input = np.matrix([[0.,0.],
                        [0.,1.],
                        [1.,0.],
                        [1.,1.]])
    result = tester.best().forward(input)
    print(result)
