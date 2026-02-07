# TensorBoard Guide for P2PFL

## What is TensorBoard?

TensorBoard is a visualization toolkit for machine learning experiments. In your P2PFL setup, it's automatically integrated through **PyTorch Lightning** to track training metrics.

## How It Works in P2PFL

### 1. Automatic Logging via PyTorch Lightning

When you train with P2PFL, PyTorch Lightning automatically creates TensorBoard logs:

```python
# In p2pfl/learning/frameworks/pytorch/lightning_learner.py
self.__trainer = Trainer(
    max_epochs=self.epochs,
    accelerator="auto",
    logger=self.logger,  # FederatedLogger - writes to TensorBoard
    enable_checkpointing=False,
    enable_model_summary=False,
    callbacks=self.callbacks.copy(),
)
```

### 2. Log Directory Structure

Your training creates logs in the `lightning_logs/` directory:

```
lightning_logs/
├── version_0/
│   ├── events.out.tfevents.XXXXX  # TensorBoard event file
│   └── hparams.yaml                # Hyperparameters
├── version_1/
│   ├── events.out.tfevents.XXXXX
│   └── hparams.yaml
├── version_2/
...
```

**Each training run creates a new version directory.**

### 3. What Gets Logged

The `FederatedLogger` (in `p2pfl/learning/frameworks/pytorch/lightning_logger.py`) logs:

- **Training loss** per epoch
- **Validation/test metrics** (accuracy, loss)
- **Hyperparameters** (learning rate, batch size, etc.)
- **Per-node metrics** (each node has its own logs)

## How to View TensorBoard

### Start TensorBoard

```bash
# View all training runs
tensorboard --logdir=lightning_logs

# View specific run
tensorboard --logdir=lightning_logs/version_0

# Specify port (default is 6006)
tensorboard --logdir=lightning_logs --port=6007
```

### Access in Browser

Open your browser and go to:
```
http://localhost:6006
```

### What You'll See

1. **SCALARS Tab**: 
   - Training/validation loss over time
   - Accuracy metrics
   - Per-node performance

2. **GRAPHS Tab**:
   - Model architecture visualization

3. **HPARAMS Tab**:
   - Hyperparameter comparison across runs

4. **TIME SERIES Tab**:
   - Metrics over training steps

## Multi-Node Training Visualization

### Problem: 1160 Version Directories!

Your `lightning_logs/` has **1160 version directories** because:
- Each node creates its own version
- Each round may create a new version
- With 30 nodes × 100 rounds = 3000+ potential versions

### Solution: Custom Logging

To better organize multi-node logs, you can:

#### Option 1: Disable PyTorch Lightning Logger

```python
# In your training script
self.__trainer = Trainer(
    max_epochs=self.epochs,
    accelerator="auto",
    logger=False,  # Disable automatic logging
    enable_checkpointing=False,
)
```

#### Option 2: Use Custom Logger Per Node

```python
from lightning.pytorch.loggers import TensorBoardLogger

# Create separate log directory per node
logger = TensorBoardLogger(
    save_dir="logs/tensorboard",
    name=f"node_{node.addr.replace(':', '_')}",
    version=f"round_{round_num}"
)
```

#### Option 3: Aggregate Logs (Recommended)

Create a script to aggregate metrics from all nodes:

```python
# aggregate_tensorboard_logs.py
from torch.utils.tensorboard import SummaryWriter
import os

def aggregate_logs(log_dir="lightning_logs", output_dir="aggregated_logs"):
    """Aggregate metrics from all nodes into single TensorBoard."""
    writer = SummaryWriter(output_dir)
    
    # Read all version directories
    for version_dir in os.listdir(log_dir):
        if not version_dir.startswith("version_"):
            continue
        
        # Parse events file and aggregate
        # ... (implementation depends on your needs)
    
    writer.close()
```

## Viewing Specific Metrics

### Filter by Node

In TensorBoard, use the search/filter:
```
# Show only node 127.0.0.1:6666
tag:127.0.0.1:6666

# Show only test metrics
tag:test_

# Show only training loss
tag:train/loss
```

### Compare Nodes

1. Select multiple runs in the left sidebar
2. TensorBoard overlays their metrics
3. Use different colors to distinguish nodes

## Custom Metrics Logging

### In Your Model

```python
# In your LightningModule (e.g., ResNetCIFAR10)
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    
    # Log to TensorBoard
    self.log('train/loss', loss, on_step=True, on_epoch=True)
    self.log('train/accuracy', accuracy, on_epoch=True)
    
    return loss

def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    
    # Log validation metrics
    self.log('val/loss', loss, on_epoch=True)
    self.log('val/accuracy', accuracy, on_epoch=True)
```

### In Your Training Script

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'logs/tensorboard/node_{node_id}')

# Log scalar
writer.add_scalar('Loss/train', loss_value, epoch)

# Log multiple scalars
writer.add_scalars('Accuracy', {
    'train': train_acc,
    'val': val_acc
}, epoch)

# Log histogram (weight distribution)
writer.add_histogram('weights/layer1', model.layer1.weight, epoch)

# Log image
writer.add_image('predictions', img_tensor, epoch)

writer.close()
```

## Cleaning Up Old Logs

### Remove Old Versions

```bash
# Keep only last 10 versions
python -c "
import os, shutil
from pathlib import Path

log_dir = Path('lightning_logs')
versions = sorted([d for d in log_dir.iterdir() if d.name.startswith('version_')], 
                  key=lambda x: int(x.name.split('_')[1]))

# Keep last 10, remove rest
for v in versions[:-10]:
    shutil.rmtree(v)
    print(f'Removed {v}')
"
```

### Automatic Cleanup

Add to your training script:

```python
import shutil
from pathlib import Path

def cleanup_old_logs(log_dir="lightning_logs", keep_last=10):
    """Keep only the last N version directories."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    versions = sorted(
        [d for d in log_path.iterdir() if d.name.startswith('version_')],
        key=lambda x: int(x.name.split('_')[1])
    )
    
    for v in versions[:-keep_last]:
        shutil.rmtree(v)
        print(f"Cleaned up {v}")

# Call before training
cleanup_old_logs()
```

## Best Practices for Multi-Node Training

### 1. Organize by Experiment

```
logs/
├── experiment_1/
│   ├── node_127.0.0.1_6666/
│   ├── node_127.0.0.1_6667/
│   └── ...
├── experiment_2/
│   └── ...
```

### 2. Log Aggregated Metrics

Instead of per-node logs, log:
- **Average loss** across all nodes
- **Min/max/std** of metrics
- **Consensus error** (deviation from mean)

### 3. Use Tags

```python
self.log('node_0/train_loss', loss)  # Per-node
self.log('global/avg_train_loss', avg_loss)  # Aggregated
self.log('global/consensus_error', std_loss)  # Variance
```

## Troubleshooting

### TensorBoard Not Showing Data

1. **Check log files exist**:
   ```bash
   ls -la lightning_logs/version_0/
   ```

2. **Verify events file**:
   ```bash
   # Should see events.out.tfevents.* file
   ```

3. **Restart TensorBoard**:
   ```bash
   # Kill existing process
   pkill -f tensorboard
   
   # Start fresh
   tensorboard --logdir=lightning_logs
   ```

### Too Many Versions

See "Cleaning Up Old Logs" section above.

### Metrics Not Updating

- TensorBoard caches data - refresh browser (Ctrl+R)
- Check if training is actually logging (add print statements)
- Verify `logger=self.logger` is set in Trainer

## Example: Viewing Your Current Logs

```bash
# Start TensorBoard
tensorboard --logdir=lightning_logs

# Open browser to http://localhost:6006

# In TensorBoard:
# 1. Go to SCALARS tab
# 2. You'll see test_loss and test_metric
# 3. Each line represents a different training run
# 4. Hover over lines to see exact values
```

## Summary

**TensorBoard in P2PFL:**
- ✅ Automatically enabled via PyTorch Lightning
- ✅ Logs stored in `lightning_logs/version_X/`
- ✅ View with `tensorboard --logdir=lightning_logs`
- ⚠️ Creates many versions with multi-node training
- 💡 Consider custom logging for better organization

**Quick Commands:**
```bash
# View logs
tensorboard --logdir=lightning_logs

# Clean old logs (keep last 10)
find lightning_logs -maxdepth 1 -name "version_*" | sort -V | head -n -10 | xargs rm -rf

# View specific experiment
tensorboard --logdir=logs/comm/run_2026-02-07_11-36-00
```
