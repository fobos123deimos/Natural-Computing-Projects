import gym

from Algoritmos_Geneticos.genetic_algorithm import Gene, Chromosome, GeneticMachine


def main():
    """
    Run a genetic algorithm to evolve action sequences for the Acrobot-v1
    environment.

    Each chromosome represents a sequence of actions.
    The chromosome score is based on the best height-related value reached
    during the simulation.
    """

    # Genetic algorithm setup
    gene_space = []
    chromosome_reward = -30
    goal_counter = 0
    chromosome_counter = 0
    goal_reached = False
    goal_history = []

    # Acrobot-v1 has three discrete actions:
    # 0 - apply negative torque
    # 1 - apply zero torque
    # 2 - apply positive torque
    #
    # The original code used range(2), which only generates actions 0 and 1.
    # Using range(3) is more appropriate for Acrobot-v1.
    for action in range(3):
        gene_space.append(action)

    gene_space = tuple(gene_space)

    # Create the genetic machine.
    #
    # If your Gene class expects an interval instead of a tuple of actions,
    # use something like:
    # Gene((0, 2), "I", 0)
    genetic_machine = GeneticMachine(
        Gene(gene_space),
        max_size=1500
    )

    # Fill the initial population with random chromosomes
    for _ in range(genetic_machine.get_size()):
        chromosome = Chromosome(
            genetic_machine.get_gene(),
            genetic_machine.get_max_size()
        )

        filled_chromosome = Chromosome.fill_random_chromosome(
            chromosome
        )

        genetic_machine.add(filled_chromosome)

    # Create and reset the Acrobot environment
    environment = gym.make("Acrobot-v1")
    environment.reset()

    print("\n")
    print(genetic_machine)
    print("\n")

    # Main evolutionary loop
    for _ in range(50):

        # Evaluate each chromosome in the current population
        for chromosome in genetic_machine.get_population():
            chromosome_counter += 1

            # Each gene represents one action in the environment
            for action in chromosome.get_string():
                environment.render()

                observation, reward, done, info = environment.step(action)

                # Acrobot observation:
                # cos(theta1), sin(theta1), cos(theta2), sin(theta2),
                # thetaDot1, thetaDot2
                cos_theta_1, sin_theta_1, cos_theta_2, sin_theta_2, theta_dot_1, theta_dot_2 = observation

                # Original height-related calculation
                horizontal_projection = (
                    cos_theta_1 * cos_theta_2
                    - sin_theta_1 * sin_theta_2
                )

                height_score = -horizontal_projection - cos_theta_1

                # Store the best score reached by this chromosome
                if height_score > chromosome_reward:
                    chromosome_reward = height_score

                # Goal condition used in the original code
                if height_score > 1:
                    goal_counter += 1
                    goal_reached = True
                    break

                if done:
                    break

            # Use the best reached score as the chromosome fitness
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
            chromosome_reward = -30
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

        # Apply selection and create the next generation
        genetic_machine.selection()

        environment.reset()

        print("\n")
        print(genetic_machine)
        print("\n")

    print("\n\n" + str(goal_history) + "\n\n")

    user_input = input("Press Enter to finish: ")

    if user_input == "\n":
        exit(0)


if __name__ == "__main__":
    main()


# Future improvement:
# Use matplotlib to plot:
# - number of goals reached per generation;
# - average chromosome score per generation;
# - best chromosome score per generation;
# - frequency of each action in the population.