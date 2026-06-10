from copy import deepcopy
from random import choice, randint, shuffle, uniform


EPSILON = 1e-12


class Gene(object):
    """
    Represents a gene configuration.

    This class supports three main uses:

    1. Discrete genes:
       Used when each gene value comes from a fixed set.
       Example: actions in Gym environments, such as (0, 1, 2).

    2. Integer or float genes:
       Used when each gene value is generated inside an interval.

    3. Neural genes:
       Used in neuroevolution, where the gene stores the neural network
       layer structure, such as (2, 3, 1).
    """

    def __init__(
        self,
        gene_space=(),
        value_type="I",
        decimal_places=0,
        network_structure=()
    ):
        """
        Initialize a gene.

        Parameters
        ----------
        gene_space : tuple
            Possible values or interval used by the gene.

        value_type : str
            Type of generated value.
            Use "I" for integer/discrete values.
            Use "F" for float values.

        decimal_places : int
            Number of decimal places used when generating float values.

        network_structure : tuple
            Neural network layer structure used in neuroevolution.
        """

        self.gene_space = gene_space
        self.value_type = value_type
        self.decimal_places = decimal_places
        self.network_structure = network_structure

    @classmethod
    def create_neural_gene(cls, layer_structure):
        """
        Create a gene for neuroevolution.

        In this case, the gene stores the neural network architecture
        instead of a regular value space.
        """

        return cls(
            gene_space=(),
            value_type="F",
            decimal_places=3,
            network_structure=layer_structure
        )

    def get_gene_space(self):
        """
        Return the gene value space.
        """

        return self.gene_space

    def get_value_type(self):
        """
        Return the gene value type.
        """

        return self.value_type

    def get_decimal_places(self):
        """
        Return the number of decimal places used for float values.
        """

        return self.decimal_places

    def get_network_structure(self):
        """
        Return the neural network layer structure.
        """

        return self.network_structure

    def set_gene_space(self, gene_space):
        """
        Update the gene value space.
        """

        self.gene_space = gene_space

    def set_network_structure(self, network_structure):
        """
        Update the neural network layer structure.
        """

        self.network_structure = network_structure

    def add_gene_value(self, gene_value):
        """
        Add a new value to the discrete gene space.
        """

        auxiliary_gene_space = list(self.gene_space)
        auxiliary_gene_space.append(gene_value)

        self.set_gene_space(tuple(auxiliary_gene_space))


class Chromosome(object):
    """
    Represents a chromosome used by the genetic algorithm.

    A chromosome stores:
    - a sequence of gene values;
    - a fitness score;
    - a mutation rate;
    - a reference to the gene configuration.
    """

    def __init__(
        self,
        gene,
        max_size=2000,
        chromosome_string=None,
        score=0,
        mutation_rate=5
    ):
        """
        Initialize a chromosome.
        """

        if chromosome_string is None:
            chromosome_string = []

        self.gene = gene
        self.max_size = max_size
        self.chromosome_string = chromosome_string
        self.score = score
        self.mutation_rate = mutation_rate

    def get_gene(self):
        """
        Return the gene configuration.
        """

        return self.gene

    def get_max_size(self):
        """
        Return the chromosome size.
        """

        return self.max_size

    def get_string(self):
        """
        Return the chromosome sequence.
        """

        return self.chromosome_string

    def get_score(self):
        """
        Return the chromosome fitness score.
        """

        return self.score

    def get_mutation_rate(self):
        """
        Return the mutation rate.
        """

        return self.mutation_rate

    def set_score(self, score):
        """
        Set the chromosome fitness score.
        """

        self.score = score

    def generate_random_value(self):
        """
        Generate one random value according to the gene configuration.

        This method is used for regular chromosomes, such as:
        - action sequences;
        - numeric chromosomes;
        - simple genetic algorithm experiments.
        """

        gene_space = self.get_gene().get_gene_space()
        value_type = self.get_gene().get_value_type()
        decimal_places = self.get_gene().get_decimal_places()

        if len(gene_space) == 0:
            raise ValueError(
                "Cannot generate a regular gene value from an empty gene space."
            )

        # Float gene generated inside an interval
        if value_type == "F":
            minimum_value = gene_space[0]
            maximum_value = gene_space[1]

            return round(
                uniform(minimum_value, maximum_value),
                decimal_places
            )

        # Integer interval, for example: (0, 2)
        if value_type == "I" and len(gene_space) == 2:
            return randint(
                gene_space[0],
                gene_space[1]
            )

        # Discrete value set, for example: (0, 1, 2)
        return choice(gene_space)

    @staticmethod
    def generate_random_weight():
        """
        Generate a random neural weight in the interval [-1.0, 1.0].
        """

        return randint(-1000, 1000) / 1000

    @classmethod
    def fill_random_chromosome(cls, chromosome):
        """
        Fill a regular chromosome with random values.
        """

        if len(chromosome.get_string()) == 0:
            chromosome_string = []

            for _ in range(chromosome.get_max_size()):
                chromosome_string.append(
                    chromosome.generate_random_value()
                )

            chromosome.chromosome_string = chromosome_string

        return chromosome

    @classmethod
    def create_neural_chromosome(
        cls,
        mutation_rate,
        chromosome_string,
        layer_structure
    ):
        """
        Create a chromosome for neuroevolution.

        The chromosome size is calculated from:

        1. All weights between consecutive layers.
        2. All bias values from hidden and output layers.
        """

        network_size = 0

        # Count all weights between consecutive layers
        for i in range(len(layer_structure) - 1):
            network_size += (
                layer_structure[i]
                * layer_structure[i + 1]
            )

        # Count all biases from hidden and output layers
        network_size += sum(
            layer_structure[1:len(layer_structure)]
        )

        return cls(
            Gene.create_neural_gene(layer_structure),
            network_size,
            chromosome_string,
            0,
            mutation_rate
        )

    @classmethod
    def fill_random_neural_chromosome(cls, chromosome):
        """
        Fill a neural chromosome with random weights and biases.

        Weight connection format:
        {
            "matrix": layer_index,
            "source_neuron": source_neuron_index,
            "target_neuron": target_neuron_index,
            "weight": weight_value
        }

        Bias connection format:
        {
            "bias_vector": layer_index,
            "neuron_index": neuron_index,
            "weight": weight_value
        }
        """

        if len(chromosome.get_string()) == 0:
            bias_connections = []
            weight_connections = []

            network_structure = (
                chromosome
                .get_gene()
                .get_network_structure()
            )

            # Iterate over each pair of consecutive layers
            for layer_index in range(len(network_structure) - 1):

                # Iterate over neurons in the next layer
                for target_neuron in range(
                    network_structure[layer_index + 1]
                ):

                    # Create one bias for each neuron in the next layer
                    bias_connections.append(
                        {
                            "bias_vector": layer_index,
                            "neuron_index": target_neuron,
                            "weight": cls.generate_random_weight()
                        }
                    )

                    # Create weights from the previous layer to this neuron
                    for source_neuron in range(
                        network_structure[layer_index]
                    ):
                        weight_connections.append(
                            {
                                "matrix": layer_index,
                                "source_neuron": source_neuron,
                                "target_neuron": target_neuron,
                                "weight": cls.generate_random_weight()
                            }
                        )

            chromosome.chromosome_string = (
                weight_connections
                + bias_connections
            )

        return chromosome

    def mutate_value(self, value):
        """
        Mutate one chromosome value.

        For neural chromosomes, only the connection weight is changed.
        For regular chromosomes, a new random value is generated.
        """

        # Neural chromosome value
        if isinstance(value, dict):
            mutated_value = deepcopy(value)
            mutated_value["weight"] = self.generate_random_weight()
            return mutated_value

        # Regular chromosome value
        return self.generate_random_value()

    def crossover(self, other_chromosome):
        """
        Perform uniform crossover between two chromosomes.

        Each child receives values randomly selected from both parents.
        Mutation may also be applied after crossover.
        """

        first_child_string = []
        second_child_string = []

        for i in range(self.get_max_size()):
            random_choice = randint(0, 1)

            if random_choice == 1:
                first_child_string.append(
                    deepcopy(self.get_string()[i])
                )

                second_child_string.append(
                    deepcopy(other_chromosome.get_string()[i])
                )

            else:
                first_child_string.append(
                    deepcopy(other_chromosome.get_string()[i])
                )

                second_child_string.append(
                    deepcopy(self.get_string()[i])
                )

        # Mutate first child
        if uniform(0, 100) <= self.get_mutation_rate():
            mutation_index = choice(
                range(self.get_max_size())
            )

            first_child_string[mutation_index] = self.mutate_value(
                first_child_string[mutation_index]
            )

        # Mutate second child
        if uniform(0, 100) <= self.get_mutation_rate():
            mutation_index = choice(
                range(self.get_max_size())
            )

            second_child_string[mutation_index] = self.mutate_value(
                second_child_string[mutation_index]
            )

        return first_child_string, second_child_string

    def __str__(self):
        """
        Return a readable representation of the chromosome.
        """

        return "\nScore: " + str(self.get_score())

    def __eq__(self, other_chromosome):
        """
        Compare chromosomes by their stored sequence.
        """

        return self.get_string() == other_chromosome.get_string()

    def __lt__(self, other_chromosome):
        """
        Compare chromosomes by score.

        This allows using max() and min() directly.
        """

        return self.get_score() < other_chromosome.get_score()


class GeneticMachine(object):
    """
    Represents the genetic algorithm engine.

    It stores the population and applies:
    - tournament selection;
    - roulette selection;
    - elitism;
    - crossover;
    - mutation;
    - generation updates.
    """

    def __init__(
        self,
        gene,
        max_size=-1,
        population=None,
        size=50,
        selection_probability=40,
        elitism=10
    ):
        """
        Initialize the genetic machine.

        Parameters
        ----------
        gene : Gene
            Gene configuration used by the population.

        max_size : int
            Chromosome size.

        population : list
            Initial population.

        size : int
            Population size.

        selection_probability : float
            Probability of selecting the best chromosome in a tournament.

        elitism : float
            Percentage of the best chromosomes preserved unchanged.
            Example: 10 means 10%.
        """

        if population is None:
            population = []

        self.gene = gene
        self.max_size = max_size
        self.population = population
        self.size = size
        self.generation = 0
        self.selection_probability = selection_probability
        self.elitism = elitism

    def get_gene(self):
        """
        Return the gene configuration.
        """

        return self.gene

    def get_max_size(self):
        """
        Return the chromosome size.
        """

        return self.max_size

    def get_population(self):
        """
        Return the current population.
        """

        return self.population

    def get_size(self):
        """
        Return the population size.
        """

        return self.size

    def get_generation(self):
        """
        Return the current generation number.
        """

        return self.generation

    def get_selection_probability(self):
        """
        Return the tournament selection probability.
        """

        return self.selection_probability

    def get_elitism(self):
        """
        Return the elitism percentage.
        """

        return self.elitism

    def set_generation(self, generation):
        """
        Set the current generation number.
        """

        self.generation = generation

    def add(self, chromosome):
        """
        Add a chromosome to the population.

        If this is the first chromosome, its size becomes the genetic
        machine's default chromosome size.
        """

        self.population.append(chromosome)

        if len(self.get_population()) == 1:
            self.max_size = self.get_population()[0].get_max_size()

    def get_elite_count(self):
        """
        Return how many chromosomes should be preserved by elitism.
        """

        elite_count = int(
            (self.get_elitism() / 100)
            * self.get_size()
        )

        return min(
            elite_count,
            len(self.get_population())
        )

    def get_elites(self):
        """
        Return the best chromosomes preserved by elitism.

        Since Chromosome.__lt__ compares by score, sorting the population
        places weaker chromosomes first and stronger chromosomes last.
        """

        elite_count = self.get_elite_count()

        if elite_count <= 0:
            return []

        sorted_population = sorted(self.get_population())

        return sorted_population[-elite_count:]

    def clone_chromosome(self, chromosome):
        """
        Create a copy of a chromosome.

        This avoids accidental reference sharing when preserving elites
        or replacing weak individuals.
        """

        return Chromosome(
            chromosome.get_gene(),
            chromosome.get_max_size(),
            deepcopy(chromosome.get_string()),
            chromosome.get_score(),
            chromosome.get_mutation_rate()
        )

    def adapt_population(self, pool_size=None):
        """
        Apply tournament selection.

        Two chromosomes are randomly selected.
        With a given probability, the best one is chosen.
        Otherwise, the worst one is chosen.
        """

        if pool_size is None:
            pool_size = self.get_size()

        selected_chromosomes = []

        selection_probability = (
            self.get_selection_probability() / 100
        )

        for _ in range(pool_size):
            first_chromosome = choice(self.get_population())
            second_chromosome = choice(self.get_population())

            random_probability = uniform(0, 1)

            if random_probability <= selection_probability:
                selected_chromosomes.append(
                    max(first_chromosome, second_chromosome)
                )
            else:
                selected_chromosomes.append(
                    min(first_chromosome, second_chromosome)
                )

        return selected_chromosomes

    def roulette_selection_index(self, inverse_score=False):
        """
        Select one chromosome index using roulette selection.

        Parameters
        ----------
        inverse_score : bool
            If False, larger scores receive larger probability.
            Use this for accuracy, reward, or positive fitness.

            If True, smaller absolute scores receive larger probability.
            Use this when your score represents error, for example:
            score = -error.
        """

        population = self.get_population()

        if len(population) == 0:
            raise ValueError("Cannot select from an empty population.")

        scores = [
            chromosome.get_score()
            for chromosome in population
        ]

        weights = []

        if inverse_score:
            # Useful when scores closer to zero are better.
            for score in scores:
                weights.append(
                    1 / (abs(score) + EPSILON)
                )
        else:
            # Useful when larger score is better.
            # Shift scores to avoid negative probabilities.
            minimum_score = min(scores)

            for score in scores:
                weights.append(
                    score - minimum_score + EPSILON
                )

        total_weight = sum(weights)

        if total_weight <= 0:
            return randint(0, len(population) - 1)

        random_number = uniform(0, total_weight)
        accumulated_weight = 0

        for index, weight in enumerate(weights):
            accumulated_weight += weight

            if random_number <= accumulated_weight:
                return index

        return len(population) - 1

    def roulette_select_chromosome(self, inverse_score=False):
        """
        Return one chromosome selected by roulette.
        """

        selected_index = self.roulette_selection_index(
            inverse_score=inverse_score
        )

        return self.get_population()[selected_index]

    def create_children_from_parents(
        self,
        parent_pool,
        number_of_children
    ):
        """
        Create children from a parent pool using crossover.
        """

        children = []

        if number_of_children <= 0:
            return children

        if len(parent_pool) == 0:
            raise ValueError("Parent pool cannot be empty.")

        while len(children) < number_of_children:
            first_parent = choice(parent_pool)
            second_parent = choice(parent_pool)

            first_child_string, second_child_string = (
                first_parent.crossover(second_parent)
            )

            first_child = Chromosome(
                self.get_gene(),
                self.get_max_size(),
                first_child_string,
                score=0,
                mutation_rate=first_parent.get_mutation_rate()
            )

            second_child = Chromosome(
                self.get_gene(),
                self.get_max_size(),
                second_child_string,
                score=0,
                mutation_rate=second_parent.get_mutation_rate()
            )

            children.append(first_child)

            if len(children) < number_of_children:
                children.append(second_child)

        return children

    def selection_by_tournament_without_elitism(self):
        """
        Apply tournament selection without elitism.

        This is close to the original behavior.
        """

        selected_chromosomes = self.adapt_population(
            pool_size=self.get_size()
        )

        new_population = self.create_children_from_parents(
            selected_chromosomes,
            self.get_size()
        )

        self.population = new_population

        self.set_generation(
            self.get_generation() + 1
        )

    def selection_by_tournament_with_elitism(self):
        """
        Apply tournament selection with elitism.

        The best chromosomes are preserved unchanged.
        The rest of the population is generated by crossover and mutation.
        """

        elites = [
            self.clone_chromosome(chromosome)
            for chromosome in self.get_elites()
        ]

        number_of_children = self.get_size() - len(elites)

        if number_of_children <= 0:
            self.population = elites[:self.get_size()]
            shuffle(self.population)

            self.set_generation(
                self.get_generation() + 1
            )

            return

        selected_chromosomes = self.adapt_population(
            pool_size=max(number_of_children, 2)
        )

        children = self.create_children_from_parents(
            selected_chromosomes,
            number_of_children
        )

        self.population = elites + children

        shuffle(self.population)

        self.set_generation(
            self.get_generation() + 1
        )

    def selection_by_roulette_with_elitism(self, inverse_score=False):
        """
        Apply roulette selection with elitism.

        Parameters
        ----------
        inverse_score : bool
            If False, larger scores have higher selection probability.
            If True, scores closer to zero have higher probability.
        """

        elites = [
            self.clone_chromosome(chromosome)
            for chromosome in self.get_elites()
        ]

        number_of_children = self.get_size() - len(elites)

        if number_of_children <= 0:
            self.population = elites[:self.get_size()]
            shuffle(self.population)

            self.set_generation(
                self.get_generation() + 1
            )

            return

        parent_pool = []

        for _ in range(max(number_of_children, 2)):
            parent_pool.append(
                self.roulette_select_chromosome(
                    inverse_score=inverse_score
                )
            )

        children = self.create_children_from_parents(
            parent_pool,
            number_of_children
        )

        self.population = elites + children

        shuffle(self.population)

        self.set_generation(
            self.get_generation() + 1
        )

    def replace_weak_individuals(self, replacement_percentage=None):
        """
        Experimental method for replacing weak chromosomes.

        This method does not perform crossover.
        It only replaces part of the weakest non-elite group with copies
        of stronger chromosomes.
        """

        if replacement_percentage is None:
            replacement_percentage = self.get_selection_probability()

        elite_count = self.get_elite_count()

        sorted_population = sorted(self.get_population())

        if elite_count > 0:
            non_elites = sorted_population[:-elite_count]
            elites = sorted_population[-elite_count:]
        else:
            non_elites = sorted_population
            elites = []

        if len(non_elites) == 0:
            return

        replacement_count = int(
            (replacement_percentage / 100)
            * len(non_elites)
        )

        if replacement_count <= 0:
            return

        strong_pool = non_elites[replacement_count:] + elites

        if len(strong_pool) == 0:
            return

        for i in range(replacement_count):
            non_elites[i] = self.clone_chromosome(
                choice(strong_pool)
            )

        self.population = non_elites + elites

        shuffle(self.population)

    def selection(self, method="tournament_elitism"):
        """
        Create a new population using one of the available selection methods.

        Available methods
        -----------------
        "tournament_elitism"
            Tournament selection with elitism.
            Recommended default for most experiments.

        "tournament"
            Tournament selection without elitism.
            Similar to the original implementation.

        "roulette"
            Roulette selection with elitism.
            Use when larger scores are better.

        "roulette_inverse"
            Roulette selection with elitism using inverse absolute score.
            Use when scores closer to zero are better, such as -error.
        """

        if method == "tournament_elitism":
            self.selection_by_tournament_with_elitism()

        elif method == "tournament":
            self.selection_by_tournament_without_elitism()

        elif method == "roulette":
            self.selection_by_roulette_with_elitism(
                inverse_score=False
            )

        elif method == "roulette_inverse":
            self.selection_by_roulette_with_elitism(
                inverse_score=True
            )

        else:
            raise ValueError(
                "Unknown selection method: " + str(method)
            )

    def __str__(self):
        """
        Return a readable representation of the genetic machine.
        """

        return "Generation: " + str(self.get_generation())


# Example usage for a discrete-action chromosome:
#
# gene = Gene((0, 1, 2))
#
# chromosome = Chromosome(
#     gene,
#     max_size=2000
# )
#
# chromosome = Chromosome.fill_random_chromosome(chromosome)
#
#
# Example usage for a neural chromosome:
#
# layer_structure = (2, 3, 1)
#
# chromosome = Chromosome.create_neural_chromosome(
#     mutation_rate=10,
#     chromosome_string=[],
#     layer_structure=layer_structure
# )
#
# chromosome = Chromosome.fill_random_neural_chromosome(chromosome)
#
#
# Selection examples:
#
# genetic_machine.selection()
# genetic_machine.selection(method="tournament")
# genetic_machine.selection(method="roulette")
# genetic_machine.selection(method="roulette_inverse")
#
#
# Use "roulette_inverse" when your score is based on error:
#
# score = -error