from __future__ import print_function

import copy
import warnings

try:
    import graphviz
except ImportError:
    graphviz = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np


def plot_stats(
    statistics,
    ylog=False,
    view=False,
    filename="avg_fitness.svg"
):
    """
    Plot the population's average fitness, fitness standard deviation,
    and best fitness across generations.

    Parameters
    ----------
    statistics : neat.StatisticsReporter
        NEAT statistics object containing fitness data for each generation.

    ylog : bool, optional
        If True, uses a symmetric logarithmic scale on the y-axis.

    view : bool, optional
        If True, displays the plot window after saving the figure.

    filename : str, optional
        Output filename used to save the plot.
    """

    if plt is None:
        warnings.warn(
            "This display is not available because matplotlib is not installed."
        )
        return

    # Generation indexes
    generations = range(len(statistics.most_fit_genomes))

    # Best genome fitness in each generation
    best_fitness = [
        genome.fitness
        for genome in statistics.most_fit_genomes
    ]

    # Average fitness and standard deviation for each generation
    average_fitness = np.array(statistics.get_fitness_mean())
    fitness_standard_deviation = np.array(
        statistics.get_fitness_stdev()
    )

    # Plot average fitness
    plt.plot(
        generations,
        average_fitness,
        "b-",
        label="Average"
    )

    # Plot one standard deviation below the average
    plt.plot(
        generations,
        average_fitness - fitness_standard_deviation,
        "g-.",
        label="-1 SD"
    )

    # Plot one standard deviation above the average
    plt.plot(
        generations,
        average_fitness + fitness_standard_deviation,
        "g-.",
        label="+1 SD"
    )

    # Plot best fitness
    plt.plot(
        generations,
        best_fitness,
        "r-",
        label="Best"
    )

    plt.title("Population's Average and Best Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    # Optional logarithmic scale
    if ylog:
        plt.gca().set_yscale("symlog")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def plot_spikes(
    spikes,
    view=False,
    filename=None,
    title=None
):
    """
    Plot spike data for a single spiking neuron.

    Each spike record is expected to contain:
    - time
    - input current
    - membrane potential
    - recovery variable
    - firing state

    Parameters
    ----------
    spikes : list of tuples
        List containing spike simulation values in the format:
        (time, current, potential, recovery, fired)

    view : bool, optional
        If True, displays the plot window.

    filename : str, optional
        If provided, saves the plot using this filename.

    title : str, optional
        Extra title information to display in the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    if plt is None:
        warnings.warn(
            "This display is not available because matplotlib is not installed."
        )
        return None

    # Separate spike data into individual lists
    time_values = [
        time
        for time, current, potential, recovery, fired in spikes
    ]

    potential_values = [
        potential
        for time, current, potential, recovery, fired in spikes
    ]

    recovery_values = [
        recovery
        for time, current, potential, recovery, fired in spikes
    ]

    current_values = [
        current
        for time, current, potential, recovery, fired in spikes
    ]

    fired_values = [
        fired
        for time, current, potential, recovery, fired in spikes
    ]

    figure = plt.figure()

    # Plot membrane potential
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mV)")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.plot(time_values, potential_values, "g-")

    if title is None:
        plt.title("Izhikevich's Spiking Neuron Model")
    else:
        plt.title(
            "Izhikevich's Spiking Neuron Model ({0!s})".format(title)
        )

    # Plot firing state
    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.plot(time_values, fired_values, "r-")

    # Plot recovery variable
    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.plot(time_values, recovery_values, "r-")

    # Plot input current
    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.plot(time_values, current_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        figure = None

    return figure


def plot_species(
    statistics,
    view=False,
    filename="speciation.svg"
):
    """
    Visualize species sizes throughout the evolutionary process.

    Parameters
    ----------
    statistics : neat.StatisticsReporter
        NEAT statistics object containing species information.

    view : bool, optional
        If True, displays the plot window after saving the figure.

    filename : str, optional
        Output filename used to save the plot.
    """

    if plt is None:
        warnings.warn(
            "This display is not available because matplotlib is not installed."
        )
        return

    # Get the size of each species for every generation
    species_sizes = statistics.get_species_sizes()

    # Number of generations processed
    number_of_generations = len(species_sizes)

    # Transpose data so each curve represents one species
    species_curves = np.array(species_sizes).T

    figure, axis = plt.subplots()

    # Stackplot shows how species occupy the population over time
    axis.stackplot(
        range(number_of_generations),
        *species_curves
    )

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close(figure)


def draw_net(
    config,
    genome,
    view=False,
    filename=None,
    node_names=None,
    show_disabled=True,
    prune_unused=False,
    node_colors=None,
    fmt="svg"
):
    """
    Draw a neural network with arbitrary topology from a NEAT genome.

    Parameters
    ----------
    config : neat.Config
        NEAT configuration object.

    genome : neat.DefaultGenome
        Genome whose neural network topology will be drawn.

    view : bool, optional
        If True, opens the rendered graph after saving.

    filename : str, optional
        Output filename used by Graphviz.

    node_names : dict, optional
        Dictionary used to rename nodes in the rendered graph.

    show_disabled : bool, optional
        If True, also displays disabled connections as dotted lines.

    prune_unused : bool, optional
        If True, removes hidden nodes that do not contribute to outputs.

    node_colors : dict, optional
        Dictionary mapping node IDs to custom colors.

    fmt : str, optional
        Graphviz output format, such as "svg", "png", or "pdf".

    Returns
    -------
    graphviz.Digraph
        The rendered Graphviz graph object.
    """

    if graphviz is None:
        warnings.warn(
            "This display is not available because graphviz is not installed."
        )
        return None

    if node_names is None:
        node_names = {}

    if node_colors is None:
        node_colors = {}

    assert isinstance(node_names, dict)
    assert isinstance(node_colors, dict)

    # Default visual attributes for hidden and output nodes
    graph_node_attributes = {
        "shape": "circle",
        "fontsize": "9",
        "height": "0.2",
        "width": "0.2"
    }

    graph = graphviz.Digraph(
        format=fmt,
        node_attr=graph_node_attributes
    )

    # Add input nodes
    input_nodes = set()

    for node_key in config.genome_config.input_keys:
        input_nodes.add(node_key)

        node_label = node_names.get(node_key, str(node_key))

        input_node_attributes = {
            "style": "filled",
            "shape": "box",
            "fillcolor": node_colors.get(node_key, "lightgray")
        }

        graph.node(
            node_label,
            _attributes=input_node_attributes
        )

    # Add output nodes
    output_nodes = set()

    for node_key in config.genome_config.output_keys:
        output_nodes.add(node_key)

        node_label = node_names.get(node_key, str(node_key))

        output_node_attributes = {
            "style": "filled",
            "fillcolor": node_colors.get(node_key, "lightblue")
        }

        graph.node(
            node_label,
            _attributes=output_node_attributes
        )

    # Optionally remove unused hidden nodes
    if prune_unused:
        active_connections = set()

        for connection_gene in genome.connections.values():
            if connection_gene.enabled or show_disabled:
                active_connections.add(
                    (
                        connection_gene.in_node_id,
                        connection_gene.out_node_id
                    )
                )

        used_nodes = copy.copy(output_nodes)
        pending_nodes = copy.copy(output_nodes)

        # Walk backward from outputs to find nodes that affect the result
        while pending_nodes:
            new_pending_nodes = set()

            for source_node, target_node in active_connections:
                if (
                    target_node in pending_nodes
                    and source_node not in used_nodes
                ):
                    new_pending_nodes.add(source_node)
                    used_nodes.add(source_node)

            pending_nodes = new_pending_nodes

    else:
        used_nodes = set(genome.nodes.keys())

    # Add hidden nodes
    for node_key in used_nodes:
        if node_key in input_nodes or node_key in output_nodes:
            continue

        hidden_node_attributes = {
            "style": "filled",
            "fillcolor": node_colors.get(node_key, "white")
        }

        graph.node(
            str(node_key),
            _attributes=hidden_node_attributes
        )

    # Add network connections
    for connection_gene in genome.connections.values():
        if connection_gene.enabled or show_disabled:
            source_key, target_key = connection_gene.key

            source_label = node_names.get(
                source_key,
                str(source_key)
            )

            target_label = node_names.get(
                target_key,
                str(target_key)
            )

            # Enabled connections are solid.
            # Disabled connections are dotted.
            edge_style = (
                "solid"
                if connection_gene.enabled
                else "dotted"
            )

            # Positive weights are green.
            # Negative weights are red.
            edge_color = (
                "green"
                if connection_gene.weight > 0
                else "red"
            )

            # Thicker lines represent stronger absolute weights.
            edge_width = str(
                0.1 + abs(connection_gene.weight / 5.0)
            )

            graph.edge(
                source_label,
                target_label,
                _attributes={
                    "style": edge_style,
                    "color": edge_color,
                    "penwidth": edge_width
                }
            )

    graph.render(filename, view=view)

    return graph