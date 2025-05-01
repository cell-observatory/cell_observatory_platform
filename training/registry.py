import copy

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def print_dependency_tree(graph: dict, indent: int = 0, root_name: str = None):
    """Recursively print the dependency graph."""
    if root_name is not None:
        print("    " * indent + f"- {root_name}")
        indent += 1
    for key, val in graph.items():
        print("    " * indent + f"- {key}")
        if isinstance(val, dict):
            print_dependency_tree(val, indent + 1)


def build_dependency_graph_and_instantiate(cfg: DictConfig) -> dict:
    """
    Recursively inspect a Hydra-style config, print the dependency graph,
    and instantiate objects in a topologically sorted order.

    This function assumes that each instantiable object is defined by a dict
    that contains a '_target_' key. The function recurses into sub-structures,
    instantiates the children first, and then instantiates the parent.
    
    Args:
        cfg (DictConfig): The nested Hydra configuration.
        
    Returns:
        A nested dictionary matching the input structure where each _target_
        dictionary is replaced by the instantiated object.
    """
    cfg_copy = copy.deepcopy(OmegaConf.to_container(cfg, resolve=True))
    
    def recursive_instantiate(node):
        # handle list or tuple nodes
        if isinstance(node, list):
            new_list = []
            for item in node:
                instantiated_item, _ = recursive_instantiate(item)
                new_list.append(instantiated_item)
            return new_list, {}
        elif isinstance(node, tuple):
            new_tuple = []
            for item in node:
                instantiated_item, _ = recursive_instantiate(item)
                new_tuple.append(instantiated_item)
            return tuple(new_tuple), {}
        
        # else if not a dict, just return node
        if not isinstance(node, dict):
            return node, {}

        dependency_graph = {}
        instantiated_children = {}

        for key, subnode in node.items():
            instantiated_subnode, child_graph = recursive_instantiate(subnode)
            instantiated_children[key] = instantiated_subnode
            # we only add to the dependency graph if the subnode was a dict
            if isinstance(subnode, dict) or isinstance(subnode, list) or isinstance(subnode, tuple):
                dependency_graph[key] = child_graph

        # if node has _target_, instantiate it
        if "_target_" in node:
            target = node["_target_"]
            print(f"Instantiating: {target} with params {instantiated_children}")
            # instantiate using the already-instantiated children as overrides
            # set _recursive_ to False to prevent further recursive instantiation
            instantiated_obj = instantiate(node, _recursive_=False, **instantiated_children)
            return instantiated_obj, dependency_graph
        else:
            return instantiated_children, dependency_graph

    instantiated_model, dependency_graph = recursive_instantiate(cfg_copy)
    print("\nDependency Graph:")
    print_dependency_tree(dependency_graph, 0, root_name="model")
    return instantiated_model