"""
Parse p2pfl log files and create aggregated metric plots
Extracts test_metric from log lines and plots mean/median across nodes
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def parse_log_file(log_file):
    """Parse log file and extract test metric per node per round"""
    data = []
    
    # Pattern to match evaluation results - support both IP addresses and node names
    # [ 2026-02-15 08:24:04 | 127.0.0.1:6666 | INFO ] 📈 Evaluated. Results: {'test_loss': 7.92, 'test_metric': 0.217, ...}
    # [ 2026-02-18 12:33:48 | node_4 | INFO ] 📈 Evaluated. Results: {'test_loss': 5.96, 'test_metric': 0.042, ...}
    pattern = r'\[ ([\d\-: ]+) \| ([^\|]+) \| INFO \] .*Evaluated\. Results: \{.*\'test_metric\': ([\d\.]+)'
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                timestamp = match.group(1)
                node = match.group(2).strip()  # Strip whitespace from node name
                metric = float(match.group(3))
                data.append({
                    'timestamp': timestamp,
                    'node': node,
                    'metric': metric
                })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return df
    
    # Add round number (each node evaluates once per round)
    df['round'] = df.groupby('node').cumcount()
    
    return df


def plot_aggregated_metric(df, output_file='log_metric_plot.png'):
    """Plot mean and median metric across all nodes per round"""
    
    if df.empty:
        print("❌ No data to plot!")
        return
    
    # Group by round and calculate statistics
    stats = df.groupby('round')['metric'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Mean and Median with error bars
    ax1.plot(stats['round'], stats['mean'], 'b-o', label='Mean', linewidth=2, markersize=8)
    ax1.plot(stats['round'], stats['median'], 'r--s', label='Median', linewidth=2, markersize=8)
    ax1.fill_between(stats['round'], 
                      stats['mean'] - stats['std'], 
                      stats['mean'] + stats['std'], 
                      alpha=0.3, label='±1 Std Dev')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Test metric', fontsize=12)
    ax1.set_title('Test metric Across All Nodes (D-SGD)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Min/Max range
    ax2.fill_between(stats['round'], stats['min'], stats['max'], alpha=0.3, label='Min-Max Range')
    ax2.plot(stats['round'], stats['mean'], 'b-o', label='Mean', linewidth=2, markersize=8)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Test metric', fontsize=12)
    ax2.set_title('Test metric Range (Min-Max)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add statistics text
    textstr = f'Nodes: {int(stats["count"].iloc[0])}\n'
    textstr += f'Rounds: {len(stats)}\n'
    textstr += f'Final Mean: {stats["mean"].iloc[-1]:.4f}\n'
    textstr += f'Final Median: {stats["median"].iloc[-1]:.4f}\n'
    textstr += f'Final Std: {stats["std"].iloc[-1]:.4f}'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_file}")
    
    # Print statistics table
    print(f"\n📊 Statistics per round:")
    print(stats.to_string(index=False))
    
    return stats


def plot_individual_nodes(df, output_file='log_individual_nodes.png'):
    """Plot individual node trajectories"""
    
    if df.empty:
        print("❌ No data to plot!")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Plot each node
    for node in df['node'].unique():
        node_data = df[df['node'] == node].sort_values('round')
        plt.plot(node_data['round'], node_data['metric'], '-o', label=node, alpha=0.7, linewidth=1.5)
    
    # Plot mean
    mean_data = df.groupby('round')['metric'].mean().reset_index()
    plt.plot(mean_data['round'], mean_data['metric'], 'k-', linewidth=3, label='Mean (All Nodes)', zorder=10)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Test metric', fontsize=12)
    plt.title('Test metric - Individual Nodes (D-SGD)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Individual nodes plot saved to {output_file}")


def export_to_csv(df, output_file='log_data.csv'):
    """Export data to CSV"""
    df.to_csv(output_file, index=False)
    print(f"✅ Data exported to {output_file}")


def main():
    import argparse
    import glob
    
    ap = argparse.ArgumentParser(description='Parse p2pfl log file(s) and plot test metric')
    ap.add_argument('--log-file', type=str, default='logs/run-2.log', help='Path to log file or pattern (e.g., logs/run-3*.log)')
    ap.add_argument('--output', type=str, default='log_metric_plot.png', help='Output plot filename')
    ap.add_argument('--export-csv', action='store_true', help='Export data to CSV')
    args = ap.parse_args()
    
    # Check if pattern contains wildcards
    if '*' in args.log_file:
        log_files = sorted(glob.glob(args.log_file))
        if not log_files:
            print(f"❌ No files found matching pattern: {args.log_file}")
            return
        print(f"📂 Reading {len(log_files)} log files:")
        for f in log_files:
            print(f"   - {f}")
    else:
        log_files = [args.log_file]
        print(f"📂 Reading log file: {args.log_file}")
    
    # Parse all log files and merge
    all_dfs = []
    for log_file in log_files:
        df = parse_log_file(log_file)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        print("❌ No evaluation results found in log file(s)!")
        print("   Make sure the log file contains lines like:")
        print("   [ ... | 127.0.0.1:6666 | INFO ] 📈 Evaluated. Results: {'test_metric': 0.217, ...}")
        return
    
    # Merge all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by timestamp to ensure correct order
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Recalculate round numbers after merging
    df['round'] = df.groupby('node').cumcount()
    
    print(f"✅ Extracted {len(df)} evaluation results")
    print(f"   Nodes: {df['node'].nunique()}")
    print(f"   Rounds per node: {df['round'].max() + 1}")
    
    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(df)
    
    # Plot aggregated metrics
    stats = plot_aggregated_metric(df, args.output)
    
    # Plot individual nodes
    if stats is not None:
        individual_output = args.output.replace('.png', '_individual.png')
        plot_individual_nodes(df, individual_output)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
