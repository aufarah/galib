# GA-Lib: Genetic Algorithm library for Python

## Introduction
GA-Lib consists of a single class named `genetic`, which has arguments: `initialPop,numGeneration,numChild,numErase,bigBest,classModel`. These arguments are tuning parameters for genetic algorithm training, and class of model that will be trained. The `genetic` class **couldn't** be used by itself because it doesn't has 2 required methods: crossover and fitness-scoring, which is customizable. The `genetic` class acts as parent class fot user-created `trainer` class. Users must make another child class as `trainer` class by calling `super` and add those methods. Methods that already contained in "genetic" are: `train`, `sort`,`best`, and `kill`, and which should be created: `crossover` and `findfit`.

Explanation about arguments in `genetic`:
* initialPop    : `(int)` How many individuals in 1st generation population (or initial population), how many elements in `population` array.
* numGeneration : `(int)` How many generations (or epochs)
* numChild : `(int)` How many new individuals created each generation
* numErase: `(int)` How many new individuals eliminated each generation, based on their fitnesss score
* bigBest: `(bool)` If True, then individual whose fitness score the biggest is the best individual. If False, then the smallest one is the best 


Explanation about methods in `genetic`:
1. `__init__(initialPop,numGeneration,numChild,numErase,bigBest,classModel)`
    <br>`__init__` receive arguments and make `initialPop`s individuals, then save it in `population` array. 
2. `sort(input,output)`
    <br>`sort` receive fitness score and population array, and sort it based on fitness score from small value to big (ascending).
3. `train()`
    <br>`train` must be called if you want to start genetic algorithm
4. `best()`
    <br> Get the best individuals


## How to Use
0. Download this repo, and place `galib.py` into your project folder
1. Import Ga-Lib, copy, and Numpy
    ```python
    import numpy as np
    import copy
    import galib
    ```
    
2. Starts with making a class of model that will be trained. This class will be used to generate individuals. For example, this is a simple neural network class named `custom`:
    ```python
    class custom:
        def __init__(self):
            self.weight = np.matrix([np.random.uniform(-10,10,2)]).T
            self.bias = np.random.uniform(-10,10,1)
        def sigmoid(self,x):
            return 1 / (1 + np.exp(-x))
        def forward(self,x):
            return self.sigmoid(x * self.weight + self.bias)
    ```
3. Make a trainer class, here i give it name `test_galib`:
    1. Declare that this class is a child from GA-Lib's `genetic` by using `super`.
    2. In `__init__`, just pass required `galib.genetic` arguments `initialPop,numGeneration,numChild,numErase,bigBest,classModel`. This means that "trainer" class just receive and forward the same arguments like `galib.genetic`.
    ```python
    class test_galib(galib.genetic):
        def __init__(self,initialPop,numGeneration,numChild,numErase,bigBest,classModel):
            super().__init__(initialPop,numGeneration,numChild,numErase,bigBest,classModel)
    ```
    3. Then, define 2 required methods: `crossover` and `findfit`. Here for crossover, i choose *randomly-weighted average* combination from 2 individuals to make a new individual (or more), and for fitness-scoring, just a simple cost function with OR operation truth table reference. Adding mutation for new individuals is also available to be implemented in `crossover`.
    <br>**notes**: output from `findfit` should be a numpy 1D array, which each element correspond to individual with the same index.
    
    ```python
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
            # fitness array length must be as same as population length, and each element correspond to individual with the same index in `population` array
            # for example, population[x] fitness score value is in fitness[x].
            
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
    ```
4. After declaring those classes, make an object from trainer class, and pass required arguments. Here i use `bigBest=False` since i used cost function for fitness scoring, and name the object `tester`.
    ```python
    tester = test_galib(initialPop=80,numGeneration=60,numChild=20,numErase=20,bigBest=False,classModel=custom)
    ```
5. Then, we can start Genetic Algoritm by calling `train`.
    ```python
    tester.train()
    ```
6. After training finished, we can get best individual by calling `best` method.
    ```python
    tester.best()
    ```
    Alternatively, you can direcly call specific individual you want by indexing them:
    ```python
    tester.population[index]
    ```
7. Voila, now you can validate your GA-trained individual (or model).
    ```python
    input = np.matrix([[0.,0.],
                        [0.,1.],
                        [1.,0.],
                        [1.,1.]])
    result = tester.population[0].forward(input)
    print(result)
    ```
    Here is the result. This trained model is totally avaiable to perform as OR logic gate:
    ```python
    [[0]
     [1]
     [1]
     [1]]
    ```
