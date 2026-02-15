# D-SGD YAML Configuration Files

Clean YAML configs for running D-SGD experiments without Python scripts.

## Available Configs

### Random Topology
- `mnist_random_dsgd.yaml` - MNIST with random topology (avg degree 5)
- `cifar10_random_dsgd.yaml` - CIFAR-10 with random topology (avg degree 5)

### Fully Connected
- `mnist_fully_dsgd.yaml` - MNIST with fully connected topology

## How to Run

### Using p2pfl CLI
```bash
p2pfl run mnist_random_dsgd.yaml
```

### Using Python module
```bash
python -m p2pfl.management.cli run mnist_random_dsgd.yaml
```

## Key Settings for D-SGD

All configs include:

```yaml
experiment:
  trainset_size: 10  # Max 10 nodes to avoid deadlock

aggregator:
  package: "p2pfl.learning.aggregators.d_sgd"
  aggregator: "DSGD"

settings:
  training:
    neighbor_only_aggregation: true  # TRUE D-SGD
    ray_actor_pool_size: 10
    aggregation_timeout: 1800  # 30 minutes
  
  heartbeat:
    timeout: 300  # 5 minutes
  
  gossip:
    exit_on_x_equal_rounds: 100  # High threshold
```

## Modifying Configs

### Change number of nodes
```yaml
network:
  nodes: 20  # Change this
```

### Change topology
```yaml
network:
  topology: "random_5"  # Options: full, star, ring, random_2, random_3, random_4, random_5
```

### Change non-IID level
```yaml
experiment:
  dataset:
    partitioning:
      params:
        alpha: 0.1  # Lower = more non-IID (0.1 is very non-IID)
```

### Change number of rounds
```yaml
experiment:
  rounds: 100  # Number of training rounds
```

## Run Variations

Test multiple configurations at once:

```bash
# Test different topologies
p2pfl run-variations mnist_random_dsgd.yaml --topologies random_3 random_4 random_5

# Test different seeds
p2pfl run-variations mnist_random_dsgd.yaml --seeds 666 777 888

# Test different alpha values
p2pfl run-variations mnist_random_dsgd.yaml --param experiment.dataset.partitioning.params.alpha=0.1,0.5,1.0

# Combine multiple variations
p2pfl run-variations mnist_random_dsgd.yaml \
  --topologies random_3 random_5 \
  --seeds 666 777 \
  --param experiment.rounds=50,100
```

## Advantages Over Python Scripts

✅ **Cleaner** - No Python code, just configuration  
✅ **Easier to modify** - Change parameters without editing code  
✅ **Reproducible** - Config files can be version controlled  
✅ **Batch experiments** - Use run-variations for parameter sweeps  
✅ **Shareable** - Easy to share exact experiment setup

## Notes

- Logs saved to `logs/` directory
- Results saved to experiment-specific directories
- Use `--help` for more options: `p2pfl run --help`
