import gym

from Algoritmos_Geneticos.genetic_algorithm import Gene, Chromosome, GeneticMachine


def main():
    """
    Run a genetic algorithm to evolve action sequences for the
    MountainCarContinuous-v0 environment.

    Each chromosome represents a sequence of continuous actions.
    The chromosome score is based on the highest position reached by the car.
    """

    # Genetic algorithm setup
    gene_space = []
    chromosome_reward = -6
    goal_counter = 0
    chromosome_counter = 0
    goal_reached = False
    goal_history = []

    # MountainCarContinuous-v0 receives continuous actions.
    # The original code uses values from -2 to 2.
    # The environment usually clips actions to the valid interval [-1, 1],
    # but using values directly inside [-1, 1] would be cleaner.
    for value in range(-2, 3):
        gene_space.append(value)

    gene_space = tuple(gene_space)

    # Create the genetic machine
    genetic_machine = GeneticMachine(
        Gene(gene_space),
        max_size=2000
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

    # Create and reset the MountainCar continuous environment
    environment = gym.make("MountainCarContinuous-v0")
    environment.reset()

    print("\n")
    print(genetic_machine)
    print("\n")

    # Main evolutionary loop
    for _ in range(50):

        # Evaluate each chromosome in the population
        for chromosome in genetic_machine.get_population():
            chromosome_counter += 1

            # Copy the chromosome action sequence
            action_sequence = list(chromosome.get_string())

            for _ in range(chromosome.get_max_size()):
                environment.render()

                # MountainCarContinuous expects the action as a list/array
                # with one value, for example: [0.5]
                current_action = [action_sequence[0]]

                observation, reward, done, info = environment.step(
                    current_action
                )

                position, velocity = observation

                # Remove the action that was just used
                del action_sequence[0]

                # Store the best position reached by this chromosome
                if position > chromosome_reward:
                    chromosome_reward = position

                # The goal position is 0.6
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