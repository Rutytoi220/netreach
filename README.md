# NetReach

A lightweight PyTorch model for exploring how attackers could move through a network.

NetReach looks at a network's security graph and estimates whether an attacker can get from one machine to another, even when the path is pretty long. It uses a recurrent neural network that can do more computation at inference time, rather than relying on a fixed number of graph-processing steps.

The idea is simple: **give the model more steps to think, and see if it can find longer paths.**

## What's the idea?

Most graph neural networks process information for a fixed number of layers. That works fine for nearby nodes, but longer paths can become difficult to detect.

NetReach takes a different approach. It keeps updating a hidden representation of the graph over multiple steps, with a few tricks to stop useful information from getting lost along the way.

You can change the number of steps at inference time (`T = 5, 10, 15, 20`) without adding more model parameters.

## How it works

At a high level, NetReach:

1. Takes a graph and a source/target pair.
2. Builds an initial representation of the graph.
3. Passes information between connected nodes over several recurrent steps.
4. Uses gating and residual connections to keep useful information around.
5. Predicts how likely the target is to be reachable from the source.

The implementation uses vectorized PyTorch operations to keep memory usage down.

### A few things worth knowing

- **Recurrent updates:** The model repeatedly updates node representations instead of stopping at a fixed graph depth.
- **Source anchoring:** The original query representation is reused at every step, so the model doesn't lose track of the starting point.
- **Variable inference depth:** Training uses different numbers of steps, while inference can use more.
- **Efficient message passing:** Batched matrix multiplication and scatter operations avoid some unnecessarily large intermediate tensors.

## Does it actually work?

I tested NetReach against a Transformer and a GRU with roughly the same parameter budget.

The models were trained on shorter paths of 3–5 hops, then evaluated on longer paths they hadn't been trained on.

Here are some of the results:

| Path length | NetReach, 5 steps | NetReach, 10 steps | NetReach, 20 steps |
|---:|---:|---:|---:|
| 3 hops | 85.0% | 91.3% | 90.7% |
| 5 hops | 82.0% | 91.0% | 91.7% |
| 8 hops | 67.0% | 84.0% | 88.7% |
| 12 hops | 54.0% | 60.0% | 75.0% |
| 16 hops | 49.7% | 51.3% | 67.3% |
| 20 hops | 45.7% | 48.7% | 55.7% |

The main takeaway: **more inference steps helped on longer paths**, especially at 12 and 16 hops. Performance still drops as paths get longer, so this definitely isn't a solved problem.

These numbers come from the current benchmark setup, not real-world penetration tests. The results should be treated as experimental until they're validated on more realistic security graphs.

### Parameter count

All three models have around 501K parameters:

| Model | Parameters |
|---|---:|
| Transformer baseline | 501,185 |
| GRU baseline | 501,017 |
| NetReach | 501,793 |

That makes the comparison fairly close in terms of model size.

## Installation

Clone the repo and install the package:

```bash
git clone https://github.com/Rutytoi220/netreach.git
cd netreach
pip install -e .
```

NetReach includes pretrained weights and supports CUDA or CPU inference.

## Quickstart

### Run an audit

Check whether a machine can reach another machine through the example security graph:

```bash
netreach audit \
  --graph examples/active_directory.json \
  --src WORKSTATION-09.CORP \
  --dst DC01.CORP \
  --depth 5 10 15 20
```

Want JSON output for another tool or script to consume?

```bash
netreach audit \
  --graph examples/active_directory.json \
  --src WORKSTATION-09.CORP \
  --dst DC01.CORP \
  --json
```

### Run the benchmark

```bash
netreach benchmark --samples 50
```

### Use it from Python

```python
from netreach import NetReachEngine

engine = NetReachEngine()

result = engine.audit(
    graph_source="examples/active_directory.json",
    source="WORKSTATION-09.CORP",
    target="DC01.CORP",
    depths=(5, 10, 15, 20),
)

print(f"Risk level: {result.risk_level}")
print(f"Probabilities: {result.probabilities}")
print(f"Path: {result.ground_truth_path}")
```

The engine automatically detects CUDA and falls back to CPU when needed.

## Supported graph formats

NetReach can work with:

- JSON edge lists
- BloodHound / Active Directory JSON
- Graphviz DOT and GV files
- GraphML
- NetworkX directed graphs

## What could it be used for?

Some possible use cases:

- **Active Directory:** Explore multi-hop relationships between domain machines, users, groups, and permissions.
- **Cloud IAM:** Study how permissions and role assumptions can create indirect access paths.
- **Network segmentation:** Check whether supposedly isolated systems are connected through other parts of the graph.

These are potential applications, not a claim that the model has already been validated for every environment.

## Running the tests

```bash
pytest tests/ -v
```

## License

NetReach is licensed under the MIT License. See [LICENSE](LICENSE) for the details.
