#!/usr/bin/env python3
"""
Monitor training progress by analyzing log files.
Run this in a separate terminal while training is running.
"""

import os
import time
import re
from collections import defaultdict
from datetime import datetime

def parse_logs(log_dir="logs"):
    """Parse log files to extract training status."""
    if not os.path.exists(log_dir):
        return None
    
    node_status = defaultdict(lambda: {"round": None, "status": "Unknown", "last_update": None})
    
    # Find most recent run directory
    run_dirs = []
    for root, dirs, files in os.walk(log_dir):
        for d in dirs:
            if d.startswith("run_"):
                run_dirs.append(os.path.join(root, d))
    
    if not run_dirs:
        return node_status
    
    latest_run = max(run_dirs, key=os.path.getmtime)
    
    # Parse each node's log file
    for filename in os.listdir(latest_run):
        if filename.endswith(".csv"):
            filepath = os.path.join(latest_run, filename)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1]
                        # Extract node address from filename
                        node_addr = filename.replace("cifar10_dcliques_node_", "").replace(".csv", "").replace("_", ":")
                        
                        # Parse last activity
                        parts = last_line.split(',')
                        if len(parts) >= 2:
                            timestamp = parts[0]
                            activity = parts[1] if len(parts) > 1 else "Unknown"
                            
                            node_status[node_addr]["last_update"] = timestamp
                            node_status[node_addr]["status"] = activity
                            
                            # Try to extract round number
                            round_match = re.search(r'round[:\s]+(\d+)', last_line, re.IGNORECASE)
                            if round_match:
                                node_status[node_addr]["round"] = int(round_match.group(1))
            except Exception as e:
                print(f"Error parsing {filename}: {e}")
    
    return node_status

def display_status(node_status):
    """Display training status in a readable format."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 80)
    print(f"Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if not node_status:
        print("No log data found. Make sure training is running.")
        return
    
    # Group by round
    rounds = defaultdict(list)
    for node, info in node_status.items():
        round_num = info["round"] if info["round"] is not None else "Unknown"
        rounds[round_num].append((node, info))
    
    # Display summary
    print(f"\nTotal Nodes: {len(node_status)}")
    print(f"Rounds in progress: {sorted([r for r in rounds.keys() if r != 'Unknown'])}")
    
    # Display by round
    for round_num in sorted(rounds.keys(), key=lambda x: x if isinstance(x, int) else -1):
        print(f"\n--- Round {round_num} ---")
        for node, info in sorted(rounds[round_num]):
            status_emoji = "✅" if "complete" in info["status"].lower() else "⏳"
            print(f"  {status_emoji} {node}: {info['status'][:50]}")
    
    # Check for stuck nodes
    print("\n--- Potential Issues ---")
    if "Unknown" in rounds and len(rounds["Unknown"]) > 0:
        print(f"⚠️  {len(rounds['Unknown'])} nodes with unknown round")
    
    round_nums = [r for r in rounds.keys() if isinstance(r, int)]
    if len(round_nums) > 1:
        print(f"⚠️  Nodes are on different rounds: {sorted(round_nums)}")
        print("   This may indicate some nodes are stuck or lagging.")

def main():
    """Main monitoring loop."""
    print("Starting training monitor...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            node_status = parse_logs()
            display_status(node_status)
            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()
