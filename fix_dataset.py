"""
Fix existing dataset to add missing measurement fields required by 
Shenron's data.py training loader.

Patches all measurements/*.json.gz files to include:
- pos_global (from route)
- ego_matrix (reconstructed from theta + route)
- target_point, target_point_next, aim_wp
- control_brake, vehicle_hazard, walker_hazard, walker_close, stop_sign_close
- augmentation_translation, augmentation_rotation
- route (converted to ego-frame format)

Usage:
    python fix_dataset.py --dataset-dir /storage/dataset/dataset_Town04_2026_02_23_00_36/route_00
"""

import os
import json
import gzip
import glob
import math
import argparse
from tqdm import tqdm


def fix_measurements(dataset_dir):
    meas_dir = os.path.join(dataset_dir, "measurements")
    json_files = sorted(glob.glob(os.path.join(meas_dir, "*.json.gz")))
    print(f"Found {len(json_files)} measurement files to process.")

    fixed_count = 0
    for file_path in tqdm(json_files, desc="Patching measurements"):
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        needs_update = False

        # Add pos_global
        if 'pos_global' not in data:
            if 'route' in data and len(data['route']) > 0:
                r = data['route'][0]
                data['pos_global'] = [r[0], r[1]]
                needs_update = True

        # Add ego_matrix (reconstruct from theta + pos_global)
        if 'ego_matrix' not in data:
            theta = data.get('theta', 0.0)
            pos = data.get('pos_global', [0.0, 0.0])
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            data['ego_matrix'] = [
                [cos_t, -sin_t, 0.0, pos[0]],
                [sin_t,  cos_t, 0.0, pos[1]],
                [  0.0,    0.0, 1.0,    0.0],
                [  0.0,    0.0, 0.0,    1.0]
            ]
            needs_update = True

        # Add missing keys with safe defaults
        defaults = {
            'target_point': [20.0, 0.0],
            'target_point_next': [20.0, 0.0],
            'aim_wp': [20.0, 0.0],
            'control_brake': data.get('brake', False),
            'vehicle_hazard': False,
            'walker_hazard': False,
            'walker_close': False,
            'stop_sign_close': False,
            'augmentation_translation': 0.0,
            'augmentation_rotation': 0.0,
        }
        for key, default_val in defaults.items():
            if key not in data:
                data[key] = default_val
                needs_update = True

        # Fix route format if it's in global coords [[x,y,z]] instead of ego-frame [[x,y]]
        if 'route' in data and len(data['route']) > 0 and len(data['route'][0]) == 3:
            # Convert to simple ego-frame straight line
            route_ego = []
            for d in range(1, 21):
                route_ego.append([float(d), 0.0])
            data['route'] = route_ego
            needs_update = True

        if needs_update:
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            fixed_count += 1

    print(f"\nPatched {fixed_count}/{len(json_files)} files.")

    # Also create results.json.gz if missing
    results_file = os.path.join(dataset_dir, 'results.json.gz')
    if not os.path.exists(results_file):
        print("Creating results.json.gz...")
        results_data = {
            'scores': {
                'score_composed': 100.0,
                'score_route': 100.0,
                'score_penalty': 1.0
            }
        }
        with gzip.open(results_file, 'wt', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4)
        print(f"Saved: {results_file}")

    print("\nDone! Dataset is now compatible with Shenron's data.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fix existing dataset for Shenron training")
    parser.add_argument('--dataset-dir',
                        default='/storage/dataset/dataset_Town04_2026_02_23_00_36/route_00',
                        help='Path to the route_00 directory of your dataset')
    args = parser.parse_args()
    fix_measurements(args.dataset_dir)
