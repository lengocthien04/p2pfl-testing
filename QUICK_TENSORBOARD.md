# Quick TensorBoard Guide - View Accuracy/Epochs

## 🚀 Quick Start (3 Steps)

### 1. Run Training
```bash
python cifar10_dcliques_dsgd_improved.py --n 10 --rounds 5
```

### 2. Start TensorBoard
```bash
tensorboard --logdir=lightning_logs
```

### 3. Open Browser
Go to: **http://localhost:6006**

---

## 📊 What You'll See

### SCALARS Tab (Main View)

You'll see 4 main metrics:

| Metric | What It Shows | Expected Behavior |
|--------|---------------|-------------------|
| **`train_accuracy`** | Training accuracy per epoch | Should increase (60-95%) |
| **`test_accuracy`** | Test accuracy per epoch | Should increase (50-75%) |
| **`train_loss`** | Training loss per epoch | Should decrease |
| **`test_loss`** | Test loss per epoch | Should decrease |

### Example View

```
📈 train_accuracy
   100% ┤                                    ╭─────
    80% ┤                          ╭────────╯
    60% ┤                 ╭───────╯
    40% ┤        ╭───────╯
    20% ┤  ─────╯
     0% ┼────────────────────────────────────────
        0    1    2    3    4    5    6    7    8
                        Epoch

📈 test_accuracy  
    75% ┤                                    ╭─────
    60% ┤                          ╭────────╯
    45% ┤                 ╭───────╯
    30% ┤        ╭───────╯
    15% ┤  ─────╯
     0% ┼────────────────────────────────────────
        0    1    2    3    4    5    6    7    8
                        Epoch
```

---

## 🎯 Key Features

### Hover for Details
Move your mouse over any point on the graph to see:
- **Epoch number**
- **Exact accuracy value**
- **Step number**

### Zoom In
- Click and drag to select a region
- Double-click to reset zoom

### Smooth Curves
- Use the **Smoothing slider** (bottom left) to reduce noise
- Useful when you have many nodes

### Compare Nodes
- Left sidebar: Select/deselect specific runs
- Each node gets a different color
- Compare how different nodes learn

---

## 🔍 Multi-Node Training

### Problem: Too Many Lines?

With 30 nodes, you'll see 30 overlapping lines. Solutions:

#### Option 1: Filter
In the search box (left sidebar), type:
```
127.0.0.1:6666
```
Shows only that node.

#### Option 2: Select Few Nodes
1. Uncheck "Show all runs"
2. Select 2-3 nodes manually
3. Compare their learning curves

#### Option 3: View Average
Run the aggregation script (see VIEW_ACCURACY_TENSORBOARD.md)

---

## 📈 What Good Training Looks Like

### Healthy Training
```
✅ train_accuracy: 15% → 80% (increasing)
✅ test_accuracy:  12% → 70% (increasing)
✅ train_loss:     2.5 → 0.5 (decreasing)
✅ test_loss:      2.7 → 0.8 (decreasing)
```

### Warning Signs
```
⚠️ test_accuracy stuck at 10% → Model not learning
⚠️ train_accuracy 95%, test_accuracy 50% → Overfitting
⚠️ Both accuracies decreasing → Learning rate too high
```

---

## 🛠️ Troubleshooting

### "No data found"
```bash
# Check logs exist
ls lightning_logs/version_0/

# Should see: events.out.tfevents.*
```

### "Metrics not showing"
```bash
# Restart TensorBoard
pkill -f tensorboard
tensorboard --logdir=lightning_logs
```

### "Too many versions"
```bash
# Clean old logs (keep last 10)
find lightning_logs -maxdepth 1 -name "version_*" | sort -V | head -n -10 | xargs rm -rf
```

---

## 💡 Pro Tips

### 1. Export Data
Click the **download icon** (top right) to export as CSV

### 2. Compare Experiments
```bash
tensorboard --logdir_spec=exp1:logs/exp1,exp2:logs/exp2
```

### 3. Real-Time Monitoring
TensorBoard auto-refreshes every 30 seconds. Keep it open while training!

### 4. Share Results
```bash
# Start on specific port
tensorboard --logdir=lightning_logs --port=6007

# Share URL with team
http://your-server-ip:6007
```

---

## 📝 Summary

**Changes Made:**
- ✅ Added `train_accuracy` logging (per epoch)
- ✅ Added `test_accuracy` logging (per epoch)
- ✅ Both logged with `on_epoch=True` for cleaner graphs

**How to View:**
1. `tensorboard --logdir=lightning_logs`
2. Open http://localhost:6006
3. Click SCALARS tab
4. See accuracy/loss curves

**What to Look For:**
- Accuracy increasing over epochs ✅
- Loss decreasing over epochs ✅
- Test accuracy < train accuracy (normal) ✅
- All nodes converging to similar accuracy ✅

---

## 🎓 Next Steps

1. **Run training** with the improved script
2. **Monitor in real-time** with TensorBoard
3. **Compare nodes** to see if they converge
4. **Export data** for further analysis
5. **Read full guide** in VIEW_ACCURACY_TENSORBOARD.md for advanced features
