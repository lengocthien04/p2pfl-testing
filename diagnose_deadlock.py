#!/usr/bin/env python3
"""
Diagnostic script to analyze training logs and identify deadlock causes.
Run this after a failed training run to understand what went wrong.
"""

import os
import re
from collections import defaultdict
from datetime import datetime

def analyze_logs(log_dir="logs/comm"):
    """Analyze communication logs to diagnose deadlock."""
    
    # Find most recent run
    if not os.path.exists(log_dir):
        print(f"❌ Log directory not found: {log_dir}")
        return
    
    run_dirs = [d for d in os.listdir(log_dir) if d.startswith("run_")]
    if not run_dirs:
        print(f"❌ No run directories found in {log_dir}")
        return
    
    latest_run = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(log_dir, d)))
    run_path = os.path.join(log_dir, latest_run)
    
    print("=" * 80)
    print(f"Analyzing: {run_path}")
    print("=" * 80)
    
    # Parse all node logs
    node_data = defaultdict(lambda: {
        "last_round": None,
        "train_set": None,
        "received_models": set(),
        "sent_models": set(),
        "heartbeat_timeouts": [],
        "grpc_errors": [],
        "aggregation_timeout": False,
        "last_activity": None
    })
    
    for filename in os.listdir(run_path):
        if not filename.endswith(".csv"):
            continue
        
        node_addr = filename.replace("cifar10_dcliques_node_", "").replace(".csv", "").replace("_", ":")
        filepath = os.path.join(run_path, filename)
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    # Extract round number
                    round_match = re.search(r'round[:\s]+(\d+)', line, re.IGNORECASE)
                    if round_match:
                        node_data[node_addr]["last_round"] = int(round_match.group(1))
                    
                    # Extract train set
                    trainset_match = re.search(r'Train set of \d+ nodes: \[(.*?)\]', line)
                    if trainset_match:
                        train_set_str = trainset_match.group(1)
                        node_data[node_addr]["train_set"] = set(
                            addr.strip().strip("'\"") for addr in train_set_str.split(",") if addr.strip()
                        )
                    
                    # Track model reception
                    if "Model received from" in line or "📦" in line:
                        source_match = re.search(r'from ([0-9.:]+)', line)
                        if source_match:
                            node_data[node_addr]["received_models"].add(source_match.group(1))
                    
                    # Track model sending
                    if "Model sent to" in line or "✅" in line:
                        dest_match = re.search(r'to ([0-9.:]+)', line)
                        if dest_match:
                            node_data[node_addr]["sent_models"].add(dest_match.group(1))
                    
                    # Track heartbeat timeouts
                    if "Heartbeat timeout" in line:
                        timeout_match = re.search(r'Heartbeat timeout for ([0-9.:]+)', line)
                        if timeout_match:
                            node_data[node_addr]["heartbeat_timeouts"].append(timeout_match.group(1))
                    
                    # Track GRPC errors
                    if "DEADLINE_EXCEEDED" in line or "Cannot send" in line:
                        node_data[node_addr]["grpc_errors"].append(line.strip())
                    
                    # Track aggregation timeout
                    if "Aggregation timeout" in line or "❌" in line:
                        node_data[node_addr]["aggregation_timeout"] = True
                    
                    # Track last activity
                    timestamp_match = re.match(r'^([0-9\-: ]+)', line)
                    if timestamp_match:
                        node_data[node_addr]["last_activity"] = timestamp_match.group(1)
        
        except Exception as e:
            print(f"⚠️  Error parsing {filename}: {e}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    # 1. Check for heartbeat timeouts (PRIMARY ISSUE)
    heartbeat_issues = {node: data for node, data in node_data.items() if data["heartbeat_timeouts"]}
    if heartbeat_issues:
        print("\n🔴 CRITICAL: HEARTBEAT TIMEOUTS DETECTED")
        print("   This is the PRIMARY cause of deadlocks in D-SGD!")
        print("   Nodes removed neighbors during training, causing train_set mismatches.\n")
        for node, data in heartbeat_issues.items():
            print(f"   {node}:")
            print(f"      Removed neighbors: {data['heartbeat_timeouts']}")
        print("\n   FIX: Increase Settings.heartbeat.TIMEOUT to 300+ seconds")
        print("        (Must be longer than training time per round)")
    else:
        print("\n✅ No heartbeat timeouts detected")
    
    # 2. Check for round mismatches
    rounds = [data["last_round"] for data in node_data.values() if data["last_round"] is not None]
    if rounds and (max(rounds) - min(rounds) > 1):
        print(f"\n⚠️  ROUND MISMATCH: Nodes on different rounds")
        print(f"   Min round: {min(rounds)}, Max round: {max(rounds)}")
        print("   This indicates some nodes are stuck while others progressed.")
    else:
        print(f"\n✅ All nodes on similar rounds (range: {min(rounds) if rounds else 'N/A'}-{max(rounds) if rounds else 'N/A'})")
    
    # 3. Check for train_set consistency
    print("\n--- Train Set Analysis ---")
    for node, data in sorted(node_data.items()):
        if data["train_set"]:
            print(f"{node}: expects models from {len(data['train_set'])} nodes")
            print(f"   Train set: {sorted(data['train_set'])}")
            print(f"   Received: {len(data['received_models'])} models from {sorted(data['received_models'])}")
            
            missing = data["train_set"] - data["received_models"] - {node}
            if missing:
                print(f"   ❌ MISSING: {sorted(missing)}")
    
    # 4. Check for GRPC errors
    grpc_issues = {node: data for node, data in node_data.items() if data["grpc_errors"]}
    if grpc_issues:
        print(f"\n⚠️  GRPC ERRORS: {len(grpc_issues)} nodes had communication errors")
        for node, data in list(grpc_issues.items())[:3]:  # Show first 3
            print(f"   {node}: {len(data['grpc_errors'])} errors")
        print("\n   FIX: Increase Settings.general.GRPC_TIMEOUT to 120+ seconds")
    else:
        print("\n✅ No GRPC errors detected")
    
    # 5. Check for aggregation timeouts
    agg_timeouts = {node: data for node, data in node_data.items() if data["aggregation_timeout"]}
    if agg_timeouts:
        print(f"\n⚠️  AGGREGATION TIMEOUTS: {len(agg_timeouts)} nodes timed out waiting for models")
        for node in list(agg_timeouts.keys())[:3]:
            print(f"   {node}")
        print("\n   This is usually a SYMPTOM of heartbeat timeouts or GRPC errors.")
    else:
        print("\n✅ No aggregation timeouts")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if heartbeat_issues:
        print("\n🔴 PRIMARY ISSUE: Heartbeat timeouts")
        print("   Nodes removed each other during training, breaking D-SGD consensus.")
        print("\n   SOLUTION:")
        print("   1. Edit p2pfl/settings.py:")
        print("      Settings.heartbeat.TIMEOUT = 300.0  # or higher")
        print("   2. Ensure AGGREGATION_TIMEOUT > HEARTBEAT_TIMEOUT")
        print("   3. Re-run training")
    elif grpc_issues:
        print("\n⚠️  PRIMARY ISSUE: GRPC communication errors")
        print("   Models failed to transmit within timeout.")
        print("\n   SOLUTION:")
        print("   1. Increase Settings.general.GRPC_TIMEOUT to 120+ seconds")
        print("   2. Check network connectivity between nodes")
    elif agg_timeouts:
        print("\n⚠️  PRIMARY ISSUE: Aggregation timeouts")
        print("   Nodes didn't receive expected models in time.")
        print("\n   SOLUTION:")
        print("   1. Check train_set consistency above")
        print("   2. Verify all nodes are connected")
        print("   3. Increase timeouts if training is legitimately slow")
    else:
        print("\n✅ No obvious issues detected")
        print("   Training may have completed successfully or logs are incomplete.")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_logs()
