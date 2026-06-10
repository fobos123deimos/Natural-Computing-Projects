import gym

from Algoritmos_Geneticos.genetic_algorithm import Gene, Chromosome, GeneticMachine


# Genetic algorithm setup
gene_space = []
chromosome_reward = -6
goal_counter = 0
chromosome_counter = 0
goal_reached = False
goal_history = []


# MountainCar-v0 has three possible discrete actions:
# 0 - push left
# 1 - no push
# 2 - push right
for action in range(3):
    gene_space.append(action)

gene_space = tuple(gene_space)


# Create the genetic machine.
#
# Important:
# This assumes that the Gene class accepts a discrete action space.
# If your Gene class expects an interval instead, use something like:
# Gene((0, 2), "I", 0)
genetic_machine = GeneticMachine(
    Gene(gene_space),
    max_size=2000,
    population=[],
    size=5
)


# Fill the initial population with random chromosomes
for _ in range(genetic_machine.get_size()):
    chromosome = Chromosome(
        genetic_machine.get_gene(),
        genetic_machine.get_max_size()
    )

    filled_chromosome = Chromosome.fill_random_chromosome(chromosome)

    genetic_machine.add(filled_chromosome)


# Create and reset the MountainCar environment
environment = gym.make("MountainCar-v0")
environment.reset()


print("\n")
print(genetic_machine)
print("\n")


# Main evolutionary loop.
#
# The original code runs only one generation.
# To run more generations, increase this value.
for _ in range(1):

    # Evaluate each chromosome in the population
    for chromosome in genetic_machine.get_population():
        chromosome_counter += 1

        # Each gene represents one action in the environment
        for action in chromosome.get_string():
            environment.render()

            observation, reward, done, info = environment.step(action)

            position, velocity = observation

            # Keep the best position reached by this chromosome
            if position > chromosome_reward:
                chromosome_reward = position

            # The goal is reached when the car reaches position 0.6
            if position == 0.6:
                goal_counter += 1
                goal_reached = True
                break

        # Use the best reached position as the chromosome score
        chromosome.set_score(chromosome_reward)

        print("\nChromosome " + str(chromosome_counter))

        if goal_reached:
            print(chromosome)
            print(genetic_machine)
            print(
                "Goal reached"
                + " // "
                + "Number of goals: "
                + str(goal_counter)
            )

        else:
            print(chromosome)
            print(genetic_machine)
            print(
                "Goal not reached"
                + " // "
                + "Number of goals: "
                + str(goal_counter)
            )

        # Reset auxiliary values before evaluating the next chromosome
        chromosome_reward = -6
        goal_reached = False
        environment.reset()

    # Store how many chromosomes reached the goal in this generation
    goal_history.append(goal_counter)

    print(
        "\n\nNumber of goals reached: "
        + str(goal_counter)
        + "\n\n"
    )

    # Reset generation counters
    goal_counter = 0
    chromosome_counter = 0

    # Apply genetic selection to create the next generation
    genetic_machine.selection()

    environment.reset()

    print("\n")
    print(genetic_machine)
    print("\n")


# Optional debug output:
#
# print("\n\n" + str(goal_history) + "\n\n")
#
# user_input = input("Press Enter to finish: ")
#
# if user_input == "\n":
#     exit(0)


# Future improvement:
# Use matplotlib to plot the average action frequencies
# or the number of goals reached per generation.