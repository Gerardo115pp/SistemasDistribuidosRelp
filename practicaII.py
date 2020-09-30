import numpy as np
import matplotlib.pyplot as plt
import json, sys, re
from random import randint, gauss

integer_list_regex = re.compile(r"^\[(-?\d+(\.\d+)?,?)+\]$") 
integer_extractor = r'(-?\d+(\.\d+)?)'

class NeuronException(Exception):
    pass

class ArgumentsException(Exception):
    pass

class Neuron:
    def __init__(self):
        self.weights = [np.random.random() for h in range(3)]
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        self.last_entry = None
        self.__learning_factor = 0.3
            
    def __isIterable(self, obj):
        try:
            iter(obj)
            return True
        except Exception:
            return False
    
    def __activationFunction(self, v):
        return v > 0
    
    def getDecisionBoundry(self):
        w1, w2 = self.weights
        x = -self.bias / w1
        y = -self.bias / w2
        
        c = -y/x
        
        xline_coords = np.array([0,x])
        yline_coords = c * xline_coords + y
        return (xline_coords, yline_coords)

    def synthesize(self, entrys):
        assert self.__isIterable(entrys)
        assert len(entrys) <= len(self.weights)
        self.last_entry = entrys
        v = np.dot(entrys, self.weights) + self.bias
        return self.__activationFunction(v)
    
    def learn(self, error):
        w_values = [self.bias] + self.weights
        x_values = [1] + self.last_entry
        for h in range(len(w_values)):
            w_values[h] += self.__learning_factor * error * x_values[h]
        self.bias = w_values[0]
        self.weights = w_values[1:]
            
            
    
    def measureError(self, error):
        pass

class Trainer:
    gates = None
    
    @classmethod
    def loadGates(cls):
        if not Trainer.gates:
            with open('gates.json') as f:
                Trainer.gates = json.load(f)
    
    def __init__(self, gate_name):
        if not Trainer.gates:
            Trainer.loadGates()
        assert gate_name in Trainer.gates.keys()
        self.__gate_name = gate_name 
        self.__target = Trainer.gates[gate_name]
        self.__neuron = Neuron()
        self.learning_data = []
    
    @property 
    def GateName(self):
        return self.__gate_name
    
    @property
    def Gate(self):
        return self.__target
    
    @property
    def Neuron(self):
        return self.__neuron
    
    def train(self, k = 0):
        got_error = False
        self.learning_data.append(self.__neuron.getDecisionBoundry())
        for entry in self.__target:
            error = self.__target[entry]['output'] - int(self.__neuron.synthesize(self.__target[entry]['x_values']))
            #print(f"error = {error}")
            if error != 0:
                got_error = True
                # self.learning_data.append(self.__neuron.getDecisionBoundry())
                self.__neuron.learn(error)
        
        if got_error:
            self.learning_data.append(self.__neuron.getDecisionBoundry())
            return self.train(k+1)

        return k


if __name__ == "__main__":
    trainer = Trainer(sys.argv[1])
    iterations = trainer.train()
    fig, ax = plt.subplots()
    plt.style.use('seaborn-pastel')
    fig.canvas.manager.full_screen_toggle()
    plt.tight_layout()
    for frame in range(len(trainer.learning_data)):
        ax.cla()
        plt.grid()
        plt.xlim([-1, 1.5])
        plt.ylim([-1, 1.5])
        for point in trainer.Gate.values():
            color = 'green' if point['output'] else 'red'
            ax.plot(point['x_values'][0], point['x_values'][1], 'bo', color=color)
        
        
        # black lines
        plt.axhline(0, color='steelblue')
        plt.axvline(0, color='steelblue')
        
        line = ax.plot(trainer.learning_data[frame][0], trainer.learning_data[frame][1], color='blue', label='Decision boundery')
        plt.legend(handles=line)
        ax.set_title(f"Took {iterations} iterations to train for '{sys.argv[1]}'")
        if frame != (len(trainer.learning_data) - 1):
            plt.pause(0.2)
    plt.show()