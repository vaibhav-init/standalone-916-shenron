#!/bin/bash
# =====================================================
#  Setup script for Shenron Standalone Agent (CARLA 0.9.16)
#  Run this from the C-Shenron project root directory
# =====================================================

# Target directory
TARGET_DIR="standalone_carla_916"
mkdir -p "${TARGET_DIR}/team_code"

echo "Copying required files..."

# 1. Core model files
cp team_code/model.py "${TARGET_DIR}/team_code/"
cp team_code/config.py "${TARGET_DIR}/team_code/"
cp team_code/transfuser.py "${TARGET_DIR}/team_code/"
cp team_code/transfuser_utils.py "${TARGET_DIR}/team_code/"
cp team_code/mask.py "${TARGET_DIR}/team_code/"
cp team_code/data.py "${TARGET_DIR}/team_code/"
cp team_code/nav_planner.py "${TARGET_DIR}/team_code/"
cp team_code/scenario_logger.py "${TARGET_DIR}/team_code/" 2>/dev/null || true

# 2. Backbone model files (resnet, timm, etc.)
cp team_code/aim.py "${TARGET_DIR}/team_code/" 2>/dev/null || true
cp team_code/bev_encoder.py "${TARGET_DIR}/team_code/" 2>/dev/null || true

# 3. Radar simulation pipeline
cp -r team_code/sim_radar_utils "${TARGET_DIR}/team_code/"
cp -r team_code/e2e_agent_sem_lidar2shenron_package "${TARGET_DIR}/team_code/"

# 4. Model weights (deploy folder)
echo ""
echo "IMPORTANT: Copy your deploy folder manually:"
echo "  cp -r logdir/shenron_radar_fb_only/deploy ${TARGET_DIR}/"
echo ""

echo "Done! Directory structure:"
find "${TARGET_DIR}" -type f | head -30
echo "..."
echo ""
echo "=== Next Steps ==="
echo "1. Copy the '${TARGET_DIR}' folder to your Python 3.12 system"
echo "2. Install dependencies:  pip install carla==0.9.16 torch torchvision opencv-python numpy scipy filterpy pynvml mat4py"
echo "3. Copy your deploy folder: cp -r logdir/shenron_radar_fb_only/deploy ${TARGET_DIR}/"
echo "4. Start CARLA 0.9.16 server"
echo "5. Run: python standalone_agent.py --model-path ./deploy --town Town04"
