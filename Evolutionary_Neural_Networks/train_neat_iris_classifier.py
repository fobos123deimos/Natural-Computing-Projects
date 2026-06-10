from __future__ import print_function
import os
import neat
import Evolutionary_Neural_Networks.neat_visualization as neat_visualization
from numpy import median, std
from random import *

inputs = []
outputs = []
X = []
Y = []
Z = []
K = []

#TRATAMENTO DATA_SET
    
file = open('iris_train_dataset.txt','r')
for linha in file:
    if linha != '\n':
        Aux = linha.split(',')
        for i in range(len(Aux)-1):
                Aux[i] = float(Aux[i])
        Aux = tuple(Aux)
        if(Aux[4][5] == 's'):
            inputs.append(Aux[0:4])
            outputs.append((1.0,0.0))
        elif(Aux[4][6] == 'i'):
            inputs.append(Aux[0:4])
            outputs.append((0.0,1.0))
        else:
            inputs.append(Aux[0:4])
            outputs.append((1.0,1.0))
       

#NORMALIZAÇÃO DOS DADOS

for In in inputs:
    X.append(In[0])
    Y.append(In[1])
    Z.append(In[2])
    K.append(In[3])
    
xm = median(X)
ym = median(Y)
zm = median(Z)
km = median(K)
xd = std(X)
yd = std(Y)
zd = std(Z)
kd = std(K)

for i in range(len(inputs)):
    Za=Zb=Zc=Zd=0
    Za = (inputs[i][0]-xm)/xd
    Zb = (inputs[i][1]-ym)/yd
    Zc = (inputs[i][2]-zm)/zd
    Zd = (inputs[i][3]-km)/kd
    inputs[i] = (Za,Zb,Zc,Zd)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        AUX = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i,o in zip(inputs,outputs):
            output = net.activate(i)
            aux = [0 if randint(0,100)/100 <= 1-i else 1 for i in output]
            aux = tuple(aux)
            if(aux == o): AUX+=1
        genome.fitness = AUX/len(inputs)
            
def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
  

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    '''
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for i,o in zip(inputs,outputs):
        output = winner_net.activate(i)
        aux = [0 if randint(0,100)/100 <= 1-i else 1 for i in output]
        aux = tuple(aux)
        print("input {!r}, expected output {!r}, got {!r}".format(i, o, aux))
    '''
    node_names = {-1:'A', -2: 'B', -3: 'C',-4: 'D',0:'a',1: 'b',2: 'c'}
    neat_visualization.draw_net(config, winner, True, node_names=node_names)
    neat_visualization.plot_stats(stats, ylog=False, view=True)
    neat_visualization.plot_species(stats, view=True)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_iris_classifier_config')
    run(config_path)

#Depois Tentar entender tanto essa parte do __main__ quando o modulo visualize, 
#e a nn dessa neat

