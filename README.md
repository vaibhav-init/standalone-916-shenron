# Shenron Standalone Agent for CARLA 0.9.16

Autonomous driving agent using **Camera + Radar** (simulated from semantic LiDAR).  
No leaderboard dependency — connects directly to CARLA.

## Quick Start

```bash
# 1. On your Python 3.12 system, install dependencies
pip install -r requirements.txt

# 2. Copy your trained model's deploy folder here
#    It should contain: model_0009.pth + config.pickle
cp -r /path/to/logdir/shenron_radar_fb_only/deploy ./

# 3. Start CARLA 0.9.16 server (separate terminal)
./CarlaUE4.sh

# 4. Run the agent
python standalone_agent.py --model-path ./deploy --town Town04 --radar-cat 1
```

## Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | *required* | Path to deploy folder |
| `--host` | localhost | CARLA server host |
| `--port` | 2000 | CARLA server port |
| `--town` | Town04 | CARLA map to load |
| `--radar-cat` | 1 | 0=front only, 1=front+back, 2=all 4 |
| `--duration` | 600 | Run duration in seconds |

## Project Structure

```
standalone_carla_916/
├── standalone_agent.py          ← Main script (run this)
├── requirements.txt             ← Python dependencies
├── deploy/                      ← Your trained model (copy here)
│   ├── model_0009.pth
│   └── config.pickle
└── team_code/                   ← Auto-copied from C-Shenron
    ├── model.py                 ← Neural network architecture
    ├── config.py                ← Model config
    ├── transfuser.py            ← TransFuser backbone
    ├── data.py                  ← Data utilities
    ├── sim_radar_utils/         ← Radar processing
    └── e2e_agent_sem_lidar2shenron_package/  ← Shenron radar sim
```

## Notes

- The agent was trained on **Town04** — best performance there
- Uses `radar_cat=1` (front + back radar) by default
- Spectator camera follows in third-person view
- Press **Ctrl+C** to stop cleanly
