from __future__ import print_function
import os
import neat
import Evolutionary_Neural_Networks.neat_visualization as neat_visualization
from numpy import median, std

inputs = []
outputs = []
X = []
Y = []

#Eu estou em DefaultReproduction!!!

#TRATAMENTO DATA_SET
    
file = open('binary_admission_dataset.txt','r')
for linha in file:
        if(linha!='\n'):
            Aux = linha[0:len(linha)-1].split(',')
            for i in range(len(Aux)):
                    Aux[i] = float(Aux[i]) 
            Aux = tuple(Aux)
            if(Aux[2] == 1.0):
                inputs.append(Aux[0:2])
                outputs.append((1.0,))
            else:
                inputs.append(Aux[0:2])
                outputs.append((-1.0,))
           


#NORMALIZAÇÃO DOS DADOS

for In in inputs:
    X.append(In[0])
    Y.append(In[1])
    
xm = median(X)
ym = median(Y)
xd = std(X)
yd = std(Y)

for i in range(len(inputs)):
    Za=Zb=0
    Za = (inputs[i][0]-xm)/xd
    Zb = (inputs[i][1]-ym)/yd
    inputs[i] = (Za,Zb)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        AUX = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i,o in zip(inputs,outputs):
            output = net.activate(i)
            if(output[0]<=0 and o[0]<0): AUX+=1
            if(output[0]>0 and o[0]>0): AUX+=1
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
    winner = p.run(eval_genomes, 150)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    '''
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for i, o in zip(inputs, outputs):
        output = winner_net.activate(i)
        print("input {!r}, expected output {!r}, got {!r}".format(i, o, output))
    '''
    node_names = {-1:'A', -2: 'B', 0:'0 ou 1'}
    neat_visualization.draw_net(config, winner, True, node_names=node_names)
    neat_visualization.plot_stats(stats, ylog=False, view=True)
    neat_visualization.plot_species(stats, view=True)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_binary_classifier_config')
    run(config_path)

#Depois Tentar entender tanto essa parte do __main__ quando o modulo visualize, 
#e a nn dessa neat
