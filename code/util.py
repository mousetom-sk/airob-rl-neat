from typing import Union, Tuple

from numpy.typing import NDArray

import neat
import graphviz

from agent import Actor
from a2c import A2C
from environment import Environment


def test_episode(policy: Union[Actor, A2C], environment: Environment) -> Tuple[float, float]:
    obs, info_init = environment.reset()
    done = False
    ret = 0

    while not done:
        act = policy.forward(obs).detach().numpy()

        obs, rew, done, info = environment.step(act)
        ret += rew

    return ret, (info_init["dist"] - info["dist"]) / info_init["dist"]

def log_results(path: str, results: NDArray) -> None:
    aggregated = [str(x)
                  for metric in zip(results.mean(axis=0), results.std(axis=0))
                  for x in metric]
    
    with open(path, "a") as out:
        print(", ".join(aggregated), file=out)

def draw_genome(
    genome: neat.DefaultGenome, config: neat.Config,
    filename=None, show_disabled=False, fmt='svg'
) -> graphviz.Digraph:
    node_names = {
        -1: "actuator x",
        -2: "actuator y",
        -3: "chip x (relative)",
        -4: "chip y (relative)",
        -5: "goal x (relative)",
        -6: "goal y (relative)",
        0: "dx",
        1: "dy"
    }
    node_colors = {}
    node_attrs = {
        "shape": "circle",
        "fontsize": "9",
        "height": "0.5",
        "width": "0.5",
        "ordering": "in"
    }

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    with dot.subgraph() as s:
        s.attr(rank="source")
        s.attr("edge", {"style": "invis"})

        nodes = []
        for k in config.genome_config.input_keys:
            inputs.add(k)
            name = node_names.get(k, str(k))
            input_attrs = {"style": "filled", "shape": "box", "fillcolor": node_colors.get(k, "lightgray")}
            
            s.node(name, _attributes=input_attrs)
            nodes.append(name)
        
        s.edges(zip(nodes, nodes[1:]))

    outputs = set()
    with dot.subgraph() as s:
        s.attr(rank="sink")
        s.attr("edge", {"style": "invis"})

        nodes = []
        for k in config.genome_config.output_keys:
            outputs.add(k)
            name = node_names.get(k, str(k))
            node_attrs = {"style": "filled", "fillcolor": node_colors.get(k, "lightblue")}

            s.node(name, _attributes=node_attrs)
            nodes.append(name)
        
        s.edges(zip(nodes, nodes[1:]))
    
    drawn_edges = set()
    active_nodes = set(outputs)
    last_num_active = 0

    while len(active_nodes) > last_num_active:
        last_num_active = len(active_nodes)

        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                input, output = cg.key

                if output not in active_nodes:
                    continue

                a = node_names.get(input, str(input))
                b = node_names.get(output, str(output))

                if a + b in drawn_edges:
                    continue

                style = "solid" if cg.enabled else "dotted"
                color = "green" if cg.weight > 0 else "red"
                width = str(0.1 + abs(cg.weight / 5.0))

                dot.edge(a, b, _attributes={"style": style, "color": color, "penwidth": width})
                
                drawn_edges.add(a + b)
                active_nodes.add(input)
    
    for n in active_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {"style": "filled", "fillcolor": node_colors.get(n, "white")}
        
        dot.node(str(n), "", _attributes=attrs)

    dot.render(filename)

    return dot
