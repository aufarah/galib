import numpy as np
import copy
    
class genetic:
    def __init__(self,initialPop,numGeneration,numChild,numErase,bigBest,classModel):
        self.population = np.array([])
        self.numGeneration = numGeneration
        self.numChild = numChild
        self.numErase = numErase
        self.initialPop = initialPop
        self.bigBest = bigBest
        for i in range(self.initialPop):
            self.population = np.append(self.population, classModel())

    def sort(self,input,output):
        for i in range(len(output)):
            lowIndex = i
            for j in range(i+1,len(output)):
                if output[j] < output[lowIndex]:
                    lowIndex = j

            output[i],output[lowIndex] = output[lowIndex],output[i]
            input[i],input[lowIndex] = input[lowIndex],input[i]
            
    def train(self):
        for generation in range(self.numGeneration):
            #mating
            self.crossover(self.population)

            #find fitness of populations
            fitness = self.findfit(self.population)

            #sorting (ascending)
            self.sort(self.population,fitness)

            #kill the unfits
            if(self.bigBest==False):
                end = len(self.population)
                deleted_elements = np.array(range(end-self.numErase,end))
            else:
                deleted_elements = np.array(range(self.numErase))
            self.population = np.delete(self.population,deleted_elements)
            fitness   = np.delete(fitness,deleted_elements)
            print(fitness[0])

    def best(self):
        if(self.bigBest==False):
            return self.population[0]
        else:
            return self.population[-1]
