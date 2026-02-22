"""
Standalone Shenron Agent for CARLA 0.9.16
Drives autonomously using Camera + Radar (simulated from semantic lidar)
No leaderboard dependency required.

Usage:
  1. Start CARLA 0.9.16 server
  2. python standalone_agent.py --model-path /path/to/deploy/folder
"""

import os
import sys
import time
import math
import argparse
import pickle
from copy import deepcopy
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import carla

# Add team_code to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM_CODE_DIR = os.path.join(SCRIPT_DIR, 'team_code')
sys.path.insert(0, TEAM_CODE_DIR)
sys.path.insert(0, os.path.join(TEAM_CODE_DIR, 'e2e_agent_sem_lidar2shenron_package'))

from model import LidarCenterNet
from config import GlobalConfig
from mask import generate_mask
from sim_radar_utils.convert2D_img import convert_sem_lidar_2D_img_func
import transfuser_utils as t_u

# UKF for GPS filtering
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

# PyTorch optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

# Radar mask
mask_for_radar = generate_mask(shape=256, start_angle=35, fov_degrees=110, end_mag=0)


def crop256X256(radar_cat):
    center_x, center_y = radar_cat.shape[1] // 2, radar_cat.shape[0] // 2
    crop_size = 256
    return radar_cat[center_y - crop_size // 2:center_y + crop_size // 2,
                     center_x - crop_size // 2:center_x + crop_size // 2]


# ============================================================
#  UKF Helper Functions (same as sensor_agent.py)
# ============================================================
def bicycle_model_forward(x, dt, steer, throttle, brake):
    """Simplified bicycle model for state prediction."""
    front_wheel_base = 1.0
    rear_wheel_base = 1.5
    steer_gain = 0.7
    brake_accel = -4.952399730682373
    throttle_accel = 0.5633837735652924

    vel = x[3]
    if brake:
        accel = brake_accel
    else:
        accel = throttle_accel * throttle

    wheel_heading_change = steer_gain * steer
    beta = math.atan(rear_wheel_base / (front_wheel_base + rear_wheel_base) * math.tan(wheel_heading_change))

    new_vel = vel + accel * dt
    new_vel = max(0.0, new_vel)
    new_heading = t_u.normalize_angle(x[2] + new_vel * math.sin(beta) / rear_wheel_base * dt)
    new_x = x[0] + new_vel * math.cos(new_heading) * dt
    new_y = x[1] + new_vel * math.sin(new_heading) * dt

    return np.array([new_x, new_y, new_heading, new_vel])


def bicycle_model_predict(x, dt, steer, throttle, brake):
    """Called by filterpy UKF per sigma point (x is a single 1D state vector)."""
    return bicycle_model_forward(x, dt, steer, throttle, brake)


def measurement_function(x):
    return x


class ShenronStandaloneAgent:
    """Standalone agent that drives using Camera + Radar in CARLA 0.9.16"""

    def __init__(self, model_path, device='cuda:0', radar_cat=1):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.radar_cat = radar_cat
        self.step = 0

        # Load config
        config_path = os.path.join(model_path, 'config.pickle')
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)

        self.config.debug = False

        # Pre-create data helper (avoid re-creating every step)
        from data import CARLA_Data
        self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

        # Load model
        print(f"Loading model from {model_path}...")
        net = LidarCenterNet(self.config)
        if self.config.sync_batch_norm:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

        # Find .pth file
        pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"No .pth file found in {model_path}")
        
        model_file = os.path.join(model_path, pth_files[0])
        print(f"Loading weights: {model_file}")
        state_dict = torch.load(model_file, map_location=self.device)
        net.load_state_dict(state_dict, strict=False)
        net.to(self.device)
        net.eval()
        self.net = net
        print("Model loaded successfully!")

        # Initialize UKF
        points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=self.residual_state_x)
        self.ukf = UKF(dim_x=4, dim_z=4, fx=bicycle_model_predict, hx=measurement_function,
                       dt=1.0 / self.config.carla_fps, points=points, x_mean_fn=self.state_mean,
                       z_mean_fn=self.z_mean, residual_x=self.residual_state_x, residual_z=self.residual_measurement)
        self.ukf.x = np.array([0, 0, 0, 0])
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        self.ukf.R = np.diag([0.5, 0.5, 0.000001, 0.000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])
        self.filter_initialized = False

        # State tracking
        self.state_log = deque(maxlen=max((self.config.lidar_seq_len * self.config.data_save_freq + 1), 20))
        self.lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq + 1)
        self.semantic_lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq + 1)

        self.lidar_last = None
        self.semantic_lidar_last = None

        self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.initialized = False

        # Stuck detection
        self.stuck_detector = 0
        self.force_move = 0

        # Target point tracking
        self.target_point_prev = np.array([0, 0])
        self.commands = deque(maxlen=2)
        self.commands.append(4)  # FOLLOW_LANE
        self.commands.append(4)

        # Uncertainty weight
        self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))

    # ============================================================
    #  UKF helper methods
    # ============================================================
    @staticmethod
    def state_mean(sigmas, Wm):
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        x[2] = math.atan2(sum_sin, sum_cos)
        x[3] = np.sum(np.dot(sigmas[:, 3], Wm))
        return x

    @staticmethod
    def z_mean(sigmas, Wm):
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        x[2] = math.atan2(sum_sin, sum_cos)
        x[3] = np.sum(np.dot(sigmas[:, 3], Wm))
        return x

    @staticmethod
    def residual_state_x(a, b):
        y = a - b
        y[2] = t_u.normalize_angle(y[2])
        return y

    @staticmethod
    def residual_measurement(a, b):
        y = a - b
        y[2] = t_u.normalize_angle(y[2])
        return y

    # ============================================================
    #  LiDAR alignment
    # ============================================================
    def align_lidar(self, lidar, x1, y1, theta1, x2, y2, theta2):
        """Aligns lidar from coordinate frame 1 to frame 2."""
        lidar = deepcopy(lidar)
        rotation = theta2 - theta1
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        delta = np.array([x1 - x2, y1 - y2])
        delta_rotated = R @ delta
        
        lidar_xy = lidar[:, :2]
        lidar[:, :2] = (R @ lidar_xy.T).T + delta_rotated
        return lidar

    def align_semantic_lidar(self, lidar, x1, y1, theta1, x2, y2, theta2):
        """Aligns semantic lidar from coordinate frame 1 to frame 2 (only xyz columns)."""
        lidar = deepcopy(lidar)
        rotation = theta2 - theta1
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        delta = np.array([x1 - x2, y1 - y2])
        delta_rotated = R @ delta
        
        lidar_xy = lidar[:, :2]
        lidar[:, :2] = (R @ lidar_xy.T).T + delta_rotated
        return lidar

    # ============================================================
    #  Process one step
    # ============================================================
    @torch.inference_mode()
    def run_step(self, rgb_image, lidar_data, semantic_lidar_data, gps, speed, compass, target_point):
        """
        Main inference step.
        
        Args:
            rgb_image: (H, W, 3) BGR camera image
            lidar_data: (N, 4) lidar points in ego frame [x, y, z, intensity]
            semantic_lidar_data: (N, 6) semantic lidar [x, y, z, cosine, index, tag]
            gps: (2,) GPS position [x, y] in CARLA coordinates
            speed: float, vehicle speed m/s
            compass: float, vehicle heading in radians
            target_point: (2,) target waypoint [x, y] in ego frame
            
        Returns:
            carla.VehicleControl
        """
        self.step += 1

        # ---- Process RGB ----
        _, compressed = cv2.imencode('.jpg', rgb_image)
        camera = cv2.imdecode(compressed, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)

        # ---- UKF Filter ----
        compass = t_u.normalize_angle(compass)
        if not self.filter_initialized:
            self.ukf.x = np.array([gps[0], gps[1], compass, speed])
            self.filter_initialized = True

        self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
        self.ukf.update(np.array([gps[0], gps[1], compass, speed]))
        filtered_state = self.ukf.x
        self.state_log.append(filtered_state)

        # ---- Command (always FOLLOW_LANE for simple driving) ----
        one_hot_command = t_u.command_to_one_hot(4)  # FOLLOW_LANE
        command = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

        # ---- Target Point ----
        ego_target = t_u.inverse_conversion_2d(target_point, filtered_state[0:2], filtered_state[2])
        ego_target = torch.from_numpy(ego_target[np.newaxis]).to(self.device, dtype=torch.float32)

        # ---- Speed ----
        gt_velocity = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)
        velocity = gt_velocity.reshape(1, 1)

        # ---- Process LiDAR ----
        if not self.initialized:
            self.lidar_last = deepcopy(lidar_data)
            self.semantic_lidar_last = deepcopy(semantic_lidar_data)
            self.initialized = True
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            return self.control

        # Align and concatenate lidar
        if len(self.state_log) >= 2:
            ego_x, ego_y, ego_theta = filtered_state[0], filtered_state[1], filtered_state[2]
            ego_x_last = self.state_log[-2][0]
            ego_y_last = self.state_log[-2][1]
            ego_theta_last = self.state_log[-2][2]

            lidar_last_aligned = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last,
                                                   ego_x, ego_y, ego_theta)
            sem_lidar_last_aligned = self.align_semantic_lidar(self.semantic_lidar_last, ego_x_last, ego_y_last,
                                                                ego_theta_last, ego_x, ego_y, ego_theta)
        else:
            lidar_last_aligned = self.lidar_last
            sem_lidar_last_aligned = self.semantic_lidar_last

        lidar_full = np.concatenate((lidar_data, lidar_last_aligned), axis=0)
        sem_lidar_full = np.concatenate((semantic_lidar_data, sem_lidar_last_aligned), axis=0)

        self.lidar_buffer.append(lidar_full)
        self.semantic_lidar_buffer.append(sem_lidar_full)

        # Wait for enough lidar frames
        if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
            self.lidar_last = deepcopy(lidar_data)
            self.semantic_lidar_last = deepcopy(semantic_lidar_data)
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
            return self.control

        # Action repeat
        if self.step % self.config.action_repeat == 1:
            self.lidar_last = deepcopy(lidar_data)
            self.semantic_lidar_last = deepcopy(semantic_lidar_data)
            return self.control

        # ---- Create LiDAR BEV + Radar ----
        lidar_indices = [i * self.config.data_save_freq for i in range(self.config.lidar_seq_len)]
        radar_list = []

        for i in lidar_indices:
            lidar_pc = deepcopy(self.lidar_buffer[i])
            lidar_histogram = torch.from_numpy(
                self.data.lidar_to_histogram_features(lidar_pc, use_ground_plane=self.config.use_ground_plane)
            ).unsqueeze(0).to(self.device, dtype=torch.float32)

            # Process radar from semantic lidar
            raw_radar = deepcopy(self.semantic_lidar_buffer[i])
            radar_np = convert_sem_lidar_2D_img_func(raw_radar, 0)

            if self.radar_cat == 1:
                radar_np_back = convert_sem_lidar_2D_img_func(raw_radar, 180)
                radar_np = radar_np * mask_for_radar
                radar_np_back = radar_np_back * mask_for_radar
                radar_np_back = np.rot90(np.rot90(radar_np_back))
                radar_np = radar_np + radar_np_back
            elif self.radar_cat == 2:
                radar_np_back = convert_sem_lidar_2D_img_func(raw_radar, 180)
                radar_np_left = convert_sem_lidar_2D_img_func(raw_radar, 270)
                radar_np_right = convert_sem_lidar_2D_img_func(raw_radar, 90)
                radar_np = radar_np * mask_for_radar
                radar_np_back = radar_np_back * mask_for_radar
                radar_np_left = radar_np_left * mask_for_radar
                radar_np_right = radar_np_right * mask_for_radar
                radar_np_left = np.rot90(radar_np_left)
                radar_np_back = np.rot90(np.rot90(radar_np_back))
                radar_np_right = np.rot90(np.rot90(np.rot90(radar_np_right)))
                radar_np = radar_np + radar_np_back + radar_np_left + radar_np_right

            radar_np_exp = np.expand_dims(radar_np, axis=2)
            radar_np_exp = np.transpose(radar_np_exp, (2, 0, 1))
            radar = torch.from_numpy(radar_np_exp).to(self.device, dtype=torch.float32).unsqueeze(0)
            radar_list.append(radar)

        radar_tensor = torch.cat(radar_list, dim=1)

        # ---- Model Forward Pass ----
        pred_wp, pred_target_speed, pred_checkpoint, \
        pred_semantic, pred_bev_semantic, pred_depth, \
        pred_bb_features, attention_weights, pred_wp_1, \
        selected_path = self.net.forward(
            rgb=rgb,
            lidar_bev=None,
            radar=radar_tensor,
            target_point=ego_target,
            ego_vel=velocity,
            command=command)

        # ---- Waypoint Selection ----
        if self.config.use_wp_gru:
            if self.config.multi_wp_output:
                if F.sigmoid(selected_path)[0].item() > 0.5:
                    pred_wp = pred_wp_1
            self.pred_wp = pred_wp

        # ---- Speed Control ----
        if self.config.use_controller_input_prediction:
            pred_target_speed_probs = F.softmax(pred_target_speed[0], dim=0)
            pred_aim_wp = pred_checkpoint[0][1].detach().cpu().numpy()
            pred_angle = -math.degrees(math.atan2(-pred_aim_wp[1], pred_aim_wp[0])) / 90.0

            if self.uncertainty_weight:
                uncertainty = pred_target_speed_probs.detach().cpu().numpy()
                if self.step % 20 == 0:  # Debug every 20 steps
                    print(f"  [DEBUG] uncertainty={uncertainty}, threshold={self.config.brake_uncertainty_threshold}")
                    print(f"  [DEBUG] target_speeds={self.config.target_speeds}, pred_angle={pred_angle:.3f}")
                if uncertainty[0] > self.config.brake_uncertainty_threshold:
                    target_speed = self.config.target_speeds[0]
                else:
                    target_speed = sum(uncertainty * self.config.target_speeds)
            else:
                idx = torch.argmax(pred_target_speed_probs)
                target_speed = self.config.target_speeds[idx]

        # ---- PID Control ----
        # Force direct controller — the waypoint PID deadlocks when car is stopped
        # because desired_speed from stationary waypoints is always near 0.
        if self.config.use_controller_input_prediction:
            steer, throttle, brake = self.net.control_pid_direct(target_speed, pred_angle, gt_velocity)
        elif self.config.use_wp_gru:
            steer, throttle, brake = self.net.control_pid(self.pred_wp, gt_velocity)
        else:
            steer, throttle, brake = 0.0, 0.0, 1.0

        if self.step % 20 == 0:
            print(f"  [DEBUG] target_speed={target_speed:.2f}, steer={steer:.3f}, throttle={throttle:.3f}, brake={brake}")

        # ---- Stuck Detection ----
        if speed < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            print(f'Agent stuck, creeping... Step: {self.step}')
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1

        self.lidar_last = deepcopy(lidar_data)
        self.semantic_lidar_last = deepcopy(semantic_lidar_data)

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        if self.step < self.config.inital_frames_delay:
            control = carla.VehicleControl(0.0, 0.0, 1.0)

        self.control = control
        return control


# ============================================================
#  CARLA 0.9.16 Main Loop
# ============================================================
def setup_carla(host='localhost', port=2000, town='Town04'):
    """Connect to CARLA and load the world."""
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.load_world(town)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    return client, world, traffic_manager


def spawn_vehicle(world):
    """Spawn the ego vehicle."""
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.lincoln.mkz_2020')[0]
    
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0]
    
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned vehicle at {spawn_point.location}")
    return vehicle


def attach_sensors(world, vehicle, config):
    """Attach camera, lidar, semantic lidar, IMU, GNSS sensors."""
    bp_lib = world.get_blueprint_library()
    sensors = {}

    # RGB Camera
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(config.camera_width))
    cam_bp.set_attribute('image_size_y', str(config.camera_height))
    cam_bp.set_attribute('fov', str(config.camera_fov))
    cam_transform = carla.Transform(
        carla.Location(x=config.camera_pos[0], y=config.camera_pos[1], z=config.camera_pos[2]),
        carla.Rotation(roll=config.camera_rot_0[0], pitch=config.camera_rot_0[1], yaw=config.camera_rot_0[2])
    )
    sensors['camera'] = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    # Regular LiDAR
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('rotation_frequency', str(config.lidar_rotation_frequency))
    lidar_bp.set_attribute('points_per_second', str(config.lidar_points_per_second))
    lidar_transform = carla.Transform(
        carla.Location(x=config.lidar_pos[0], y=config.lidar_pos[1], z=config.lidar_pos[2]),
        carla.Rotation(roll=config.lidar_rot[0], pitch=config.lidar_rot[1], yaw=config.lidar_rot[2])
    )
    sensors['lidar'] = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    # Semantic LiDAR (for radar simulation)
    sem_lidar_bp = bp_lib.find('sensor.lidar.ray_cast_semantic')
    sem_lidar_bp.set_attribute('rotation_frequency', str(config.lidar_rotation_frequency))
    sem_lidar_bp.set_attribute('points_per_second', str(config.lidar_points_per_second))
    sensors['semantic_lidar'] = world.spawn_actor(sem_lidar_bp, lidar_transform, attach_to=vehicle)

    # IMU
    imu_bp = bp_lib.find('sensor.other.imu')
    sensors['imu'] = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)

    # GNSS
    gnss_bp = bp_lib.find('sensor.other.gnss')
    sensors['gnss'] = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)

    return sensors


class SensorData:
    """Collects sensor data callbacks."""
    def __init__(self):
        self.rgb = None
        self.lidar = None
        self.semantic_lidar = None
        self.imu = None
        self.gnss = None

    def on_rgb(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Drop alpha
        self.rgb = array

    def on_lidar(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        # Convert to ego coordinate frame (CARLA lidar is already in sensor frame)
        self.lidar = points

    def on_semantic_lidar(self, data):
        # Semantic lidar has: x, y, z, cos_angle, object_index, semantic_tag
        points = np.frombuffer(data.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
        ]))
        result = np.column_stack([
            points['x'], points['y'], points['z'],
            points['CosAngle'],
            points['ObjIdx'].astype(np.float64),
            points['ObjTag'].astype(np.float64)
        ])
        self.semantic_lidar = result

    def on_imu(self, data):
        self.imu = data

    def on_gnss(self, data):
        self.gnss = data


def spawn_obstacle_ahead(world, ego_vehicle, distance=40.0):
    """Spawn a stopped vehicle in the ego's lane, `distance` meters ahead."""
    import random
    map = world.get_map()
    ego_transform = ego_vehicle.get_transform()
    ego_wp = map.get_waypoint(ego_transform.location)

    # Walk forward along the lane
    next_wps = ego_wp.next(distance)
    if not next_wps:
        print("Could not find waypoint ahead for obstacle!")
        return None
    obstacle_wp = next_wps[0]

    bp_lib = world.get_blueprint_library()
    # Pick a big vehicle so it's clearly visible
    obstacle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    obstacle_bp.set_attribute('color', '255,0,0')  # Red so it's obvious

    obstacle_transform = obstacle_wp.transform
    obstacle_transform.location.z += 0.5  # Lift slightly to avoid ground clipping

    obstacle = world.spawn_actor(obstacle_bp, obstacle_transform)
    # Apply handbrake so it stays put
    obstacle.apply_control(carla.VehicleControl(hand_brake=True))
    print(f"  Spawned STOPPED obstacle {distance}m ahead at {obstacle_transform.location}")
    return obstacle


def spawn_traffic(client, world, tm, num_vehicles=30, num_walkers=20):
    """Spawn NPC vehicles and pedestrians."""
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_actors = []
    walker_actors = []

    # --- Spawn Vehicles ---
    vehicle_bps = bp_lib.filter('vehicle.*')
    # Filter out bikes/motorcycles for more realistic traffic
    vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) >= 4]

    import random
    random.shuffle(spawn_points)
    num_vehicles = min(num_vehicles, len(spawn_points) - 1)  # Reserve one for ego

    batch = []
    for i in range(num_vehicles):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        bp.set_attribute('role_name', 'autopilot')
        batch.append(carla.command.SpawnActor(bp, spawn_points[i + 1]).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))

    results = client.apply_batch_sync(batch, True)
    for result in results:
        if not result.error:
            vehicle_actors.append(result.actor_id)

    print(f"Spawned {len(vehicle_actors)} NPC vehicles")

    # --- Spawn Walkers ---
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    walker_controller_bp = bp_lib.find('controller.ai.walker')

    walker_ids = []
    controller_ids = []

    for _ in range(num_walkers):
        spawn_loc = world.get_random_location_from_navigation()
        if spawn_loc is None:
            continue
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        try:
            walker = world.spawn_actor(bp, carla.Transform(spawn_loc))
            walker_ids.append(walker.id)
            controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
            controller_ids.append(controller.id)
        except:
            pass

    # Wait a tick then start walking
    world.tick()
    all_controllers = world.get_actors(controller_ids)
    for controller in all_controllers:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1.0 + random.random() * 1.5)  # 1-2.5 m/s

    print(f"Spawned {len(walker_ids)} pedestrians")

    walker_actors = walker_ids + controller_ids
    return vehicle_actors, walker_actors


def main():
    parser = argparse.ArgumentParser(description='Shenron Standalone Agent for CARLA 0.9.16')
    parser.add_argument('--model-path', required=True, help='Path to deploy folder with model .pth and config.pickle')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', default='Town04')
    parser.add_argument('--radar-cat', type=int, default=1, help='0=front, 1=front+back, 2=all 4 directions')
    parser.add_argument('--duration', type=int, default=600, help='Duration in seconds')
    parser.add_argument('--vehicles', type=int, default=30, help='Number of NPC vehicles')
    parser.add_argument('--walkers', type=int, default=20, help='Number of pedestrians')
    args = parser.parse_args()

    # Set environment variables
    os.environ['RADAR_CAT'] = str(args.radar_cat)
    os.environ['RADAR_CHANNEL'] = '1'
    os.environ['UNCERTAINTY_WEIGHT'] = '1'

    actors = []
    try:
        # 1. Connect to CARLA
        print("Connecting to CARLA...")
        client, world, tm = setup_carla(args.host, args.port, args.town)
        print(f"Connected! Town: {args.town}")

        # 2. Load agent
        agent = ShenronStandaloneAgent(args.model_path, radar_cat=args.radar_cat)

        # 3. Spawn vehicle
        vehicle = spawn_vehicle(world)
        actors.append(vehicle)

        # 4. Attach sensors
        sensor_data = SensorData()
        sensors = attach_sensors(world, vehicle, agent.config)
        sensors['camera'].listen(sensor_data.on_rgb)
        sensors['lidar'].listen(sensor_data.on_lidar)
        sensors['semantic_lidar'].listen(sensor_data.on_semantic_lidar)
        sensors['imu'].listen(sensor_data.on_imu)
        sensors['gnss'].listen(sensor_data.on_gnss)
        actors.extend(sensors.values())

        # 5. Set spectator (third-person view)
        spectator = world.get_spectator()

        # Wait for sensors to initialize
        for _ in range(10):
            world.tick()

        # 5b. Spawn a stopped obstacle in the ego's lane
        obstacle = spawn_obstacle_ahead(world, vehicle, distance=40.0)
        if obstacle:
            actors.append(obstacle)

        # 5c. Spawn traffic (optional — uncomment if you also want moving NPCs)
        # npc_vehicle_ids, walker_ids = spawn_traffic(client, world, tm, args.vehicles, args.walkers)
        npc_vehicle_ids, walker_ids = [], []

        # 6. Get the route's next waypoint for target_point
        map = world.get_map()

        print("\n" + "=" * 50)
        print("  Agent is driving! Press Ctrl+C to stop.")
        print("=" * 50 + "\n")

        # 7. Main loop
        start_time = time.time()
        fps_counter = 0
        fps_timer = time.time()

        while time.time() - start_time < args.duration:
            world.tick()

            if sensor_data.rgb is None or sensor_data.lidar is None or sensor_data.semantic_lidar is None:
                continue

            # Get vehicle state
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # GPS position (CARLA coordinates)
            gps = np.array([transform.location.x, transform.location.y])
            compass = math.radians(transform.rotation.yaw)

            # Get target point (next waypoint ~20m ahead)
            current_wp = map.get_waypoint(transform.location)
            target_wps = current_wp.next(20.0)
            if target_wps:
                target_loc = target_wps[0].transform.location
                target_point = np.array([target_loc.x, target_loc.y])
            else:
                target_point = gps + np.array([20.0, 0.0])

            # Run agent
            control = agent.run_step(
                rgb_image=sensor_data.rgb,
                lidar_data=sensor_data.lidar,
                semantic_lidar_data=sensor_data.semantic_lidar,
                gps=gps,
                speed=speed,
                compass=compass,
                target_point=target_point
            )

            vehicle.apply_control(control)

            # Update spectator (third-person view)
            loc = transform.transform(carla.Location(x=-6.0, z=3.0))
            spectator.set_transform(carla.Transform(loc, carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw)))

            # FPS counter
            fps_counter += 1
            if time.time() - fps_timer > 5.0:
                fps = fps_counter / (time.time() - fps_timer)
                print(f"Speed: {speed:.1f} m/s | Steer: {control.steer:.2f} | "
                      f"Throttle: {control.throttle:.2f} | Brake: {control.brake:.2f} | FPS: {fps:.1f}")
                fps_counter = 0
                fps_timer = time.time()

    except KeyboardInterrupt:
        print("\nStopping agent...")
    finally:
        print("Cleaning up...")
        # Destroy NPC traffic
        try:
            client.apply_batch([carla.command.DestroyActor(x) for x in npc_vehicle_ids])
            client.apply_batch([carla.command.DestroyActor(x) for x in walker_ids])
        except:
            pass
        for actor in reversed(actors):
            actor.destroy()
        
        # Restore async mode
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except:
            pass
        print("Done!")


if __name__ == '__main__':
    main()
