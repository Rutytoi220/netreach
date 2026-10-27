"""
Graph Loader Adapter for NetReach.

Supports loading network security graphs from:
- JSON edge lists: [{"source": "u", "target": "v", "relation": "AdminTo"}]
- BloodHound / Network JSON formats
- Standard GraphML (.graphml)
- Graphviz DOT (.dot, .gv)
- Existing NetworkX DiGraph instances
"""

import os
import re
import json
from typing import Union, List, Dict, Any
import networkx as nx


def parse_dot_string(dot_text: str) -> nx.DiGraph:
    """Extract directed edges from Graphviz DOT format without requiring external C libraries."""
    G = nx.DiGraph()
    # Matches 'nodeA -> nodeB' allowing quotes and identifier characters
    edge_pattern = re.compile(
        r'["\']?([a-zA-Z0-9_\-\.\:\@\$\\/]+)["\']?\s*->\s*["\']?([a-zA-Z0-9_\-\.\:\@\$\\/]+)["\']?'
    )
    attr_pattern = re.compile(r'\[([^\]]+)\]')

    for line in dot_text.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        m = edge_pattern.search(line)
        if m:
            u, v = m.group(1), m.group(2)
            attrs = {}
            attr_match = attr_pattern.search(line)
            if attr_match:
                attr_str = attr_match.group(1)
                # simple key=value parsing
                for item in re.split(r'[,;]', attr_str):
                    if "=" in item:
                        k, val = item.split("=", 1)
                        attrs[k.strip()] = val.strip().strip('"\'')
            G.add_edge(u, v, **attrs)
    return G


def load_graph(source: Union[str, nx.DiGraph, List[Dict[str, Any]], Dict[str, Any]]) -> nx.DiGraph:
    """
    Load a graph into a NetworkX DiGraph.

    Args:
        source: File path (str), NetworkX DiGraph, or Python dict/list.
    Returns:
        nx.DiGraph: Directed graph with string node identifiers.
    """
    if isinstance(source, nx.DiGraph):
        return source

    if isinstance(source, nx.Graph):
        # Convert undirected to directed
        return source.to_directed()

    if isinstance(source, list):
        # List of edge dicts: [{"source": "u", "target": "v", ...}]
        G = nx.DiGraph()
        for item in source:
            u = str(item.get("source") or item.get("src") or item.get("from"))
            v = str(item.get("target") or item.get("dst") or item.get("to"))
            attrs = {k: v for k, v in item.items() if k not in ("source", "src", "from", "target", "dst", "to")}
            G.add_edge(u, v, **attrs)
        return G

    if isinstance(source, dict):
        # Dict with 'edges' or 'links' or 'data'
        edges_list = source.get("edges") or source.get("links") or source.get("data")
        if edges_list and isinstance(edges_list, list):
            return load_graph(edges_list)
        # Check if node-link format
        try:
            return nx.node_link_graph(source, directed=True)
        except Exception:
            pass
        raise ValueError("Unsupported dictionary format for graph loading.")

    if isinstance(source, str):
        if not os.path.exists(source):
            # Check if it's raw DOT string
            if "->" in source:
                return parse_dot_string(source)
            raise FileNotFoundError(f"Graph file not found: {source}")

        ext = os.path.splitext(source)[1].lower()

        if ext in (".json", ".js"):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            return load_graph(data)

        if ext == ".graphml":
            G = nx.read_graphml(source)
            return G if isinstance(G, nx.DiGraph) else G.to_directed()

        if ext in (".dot", ".gv"):
            with open(source, "r", encoding="utf-8") as f:
                text = f.read()
            return parse_dot_string(text)

        # Fallback to text parsing (line-separated u -> v or u \t v)
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
        if "->" in content:
            return parse_dot_string(content)

        # Try edge list
        try:
            return nx.read_edgelist(source, create_using=nx.DiGraph)
        except Exception as e:
            raise ValueError(f"Unable to parse graph from {source}: {e}")

    raise TypeError(f"Unsupported graph source type: {type(source)}")
