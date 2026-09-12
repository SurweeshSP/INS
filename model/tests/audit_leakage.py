import os
import sys
sys.path.append(r"C:\Users\surwe\Project\INS\model")

import pandas as pd
import numpy as np
from collections import Counter

def main():
    sm_dir = r"C:\Users\surwe\Project\INS\IO-VNBD\preprocess_data"
    vm_root = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
    
    # 1. Find all pairs
    s_files = [f for f in os.listdir(sm_dir) if f.startswith("S-") and f.endswith(".csv")]
    
    v_paths = {}
    for dirpath, dirnames, filenames in os.walk(vm_root):
        for f in filenames:
            if f.startswith("V-") and f.endswith(".csv"):
                name = f[2:-4]
                v_paths[name.lower()] = os.path.join(dirpath, f)
                
    pairs = []
    for sf in s_files:
        name = sf[2:-4]
        key = name.lower()
        if key in v_paths:
            pairs.append({
                'name': name,
                's_path': os.path.join(sm_dir, sf),
                'v_path': v_paths[key]
            })
        else:
            matched = False
            for k, p in v_paths.items():
                if k in key or key in k:
                    pairs.append({
                        'name': name,
                        's_path': os.path.join(sm_dir, sf),
                        'v_path': p
                    })
                    matched = True
                    break
                    
    print(f"Matched {len(pairs)} drive sessions.")
    
    # Let's split pairs by driver/recording to prevent leakage
    # We will partition at the recording level:
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    for pair in pairs:
        name = pair['name']
        # Partition rule:
        # Driver A runs (S*) -> Train
        # Driver B run (M) -> Train
        # Driver D runs (Y*) -> Test
        # Driver E runs:
        #   Vfa* -> Test
        #   Vta1-Vta20 -> Train
        #   Vta21-Vta30 -> Val
        #   Vtb1-Vtb8 -> Train
        #   Vtb9-Vtb13 -> Val
        #   Vw1-Vw10 -> Train
        #   Vw11-Vw14a -> Val
        #   Vw14b-Vw17 -> Test
        
        if name.startswith('S') and name != 'S-M': # Driver A (S1, S2, etc.)
            train_pairs.append(pair)
        elif name == 'M': # Driver B
            train_pairs.append(pair)
        elif name.startswith('Y'): # Driver D
            test_pairs.append(pair)
        elif name.startswith('Vfa'): # Driver E (Vfa)
            test_pairs.append(pair)
        elif name.startswith('Vta'):
            # Extract number
            num_str = ''.join([c for c in name if c.isdigit()])
            if num_str:
                num = int(num_str)
                if num <= 20:
                    train_pairs.append(pair)
                else:
                    val_pairs.append(pair)
            else:
                train_pairs.append(pair)
        elif name.startswith('Vtb'):
            num_str = ''.join([c for c in name if c.isdigit()])
            if num_str:
                num = int(num_str)
                if num <= 8:
                    train_pairs.append(pair)
                else:
                    val_pairs.append(pair)
            else:
                train_pairs.append(pair)
        elif name.startswith('Vw'):
            num_str = ''.join([c for c in name if c.isdigit()])
            if num_str:
                num = int(num_str)
                if num <= 10:
                    train_pairs.append(pair)
                elif num <= 14:
                    val_pairs.append(pair)
                else:
                    test_pairs.append(pair)
            else:
                train_pairs.append(pair)
        else:
            train_pairs.append(pair)
            
    print(f"\nTrajectory split counts:")
    print(f"  Train drives: {len(train_pairs)}")
    print(f"  Val drives:   {len(val_pairs)}")
    print(f"  Test drives:  {len(test_pairs)}")
    
    # Check window counts for train
    # Let's count approximate rows
    total_train_rows = sum(len(pd.read_csv(p['s_path'])) for p in train_pairs)
    total_val_rows = sum(len(pd.read_csv(p['s_path'])) for p in val_pairs)
    total_test_rows = sum(len(pd.read_csv(p['s_path'])) for p in test_pairs)
    
    print(f"\nApproximate total rows:")
    print(f"  Train: {total_train_rows:,} rows")
    print(f"  Val:   {total_val_rows:,} rows")
    print(f"  Test:  {total_test_rows:,} rows")

if __name__ == '__main__':
    main()
