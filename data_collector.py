"""
Data Collection Script for CARLA 0.9.16
Drives autonomously using CARLA Traffic Manager Autopilot and saves sensor logs
compatible with Shenron / C-Shenron training.
"""

import os
import sys
import time
import math
import argparse
import datetime
import gzip
import json

import cv2
import numpy as np
import laspy

import carla

# Add team_code to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM_CODE_DIR = os.path.join(SCRIPT_DIR, 'team_code')
sys.path.insert(0, TEAM_CODE_DIR)
sys.path.insert(0, os.path.join(TEAM_CODE_DIR, 'e2e_agent_sem_lidar2shenron_package'))

from mask import generate_mask
from sim_radar_utils.convert2D_img import convert_sem_lidar_2D_img_func

# Radar mask
mask_for_radar = generate_mask(shape=256, start_angle=35, fov_degrees=110, end_mag=0)

# ============================================================
#  Data Collector Setup
# ============================================================
class SimpleConfig:
    """Minimal config required for sensors"""
    camera_width = 1024
    camera_height = 256
    camera_fov = 110
    camera_pos = [1.3, 0.0, 2.3]
    camera_rot_0 = [0.0, 0.0, 0.0]
    lidar_pos = [1.3, 0.0, 2.5]
    lidar_rot = [0.0, 0.0, -90.0]
    lidar_rotation_frequency = 20
    lidar_points_per_second = 600000

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
    print(f"Spawned auto-pilot vehicle at {spawn_point.location}")
    return vehicle

def attach_sensors(world, vehicle, config):
    """Attach camera, lidar, semantic lidar sensors."""
    bp_lib = world.get_blueprint_library()
    sensors = {}

    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(config.camera_width))
    cam_bp.set_attribute('image_size_y', str(config.camera_height))
    cam_bp.set_attribute('fov', str(config.camera_fov))
    cam_transform = carla.Transform(
        carla.Location(x=config.camera_pos[0], y=config.camera_pos[1], z=config.camera_pos[2]),
        carla.Rotation(roll=config.camera_rot_0[0], pitch=config.camera_rot_0[1], yaw=config.camera_rot_0[2])
    )
    sensors['camera'] = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('rotation_frequency', str(config.lidar_rotation_frequency))
    lidar_bp.set_attribute('points_per_second', str(config.lidar_points_per_second))
    lidar_transform = carla.Transform(
        carla.Location(x=config.lidar_pos[0], y=config.lidar_pos[1], z=config.lidar_pos[2]),
        carla.Rotation(roll=config.lidar_rot[0], pitch=config.lidar_rot[1], yaw=config.lidar_rot[2])
    )
    sensors['lidar'] = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    sem_lidar_bp = bp_lib.find('sensor.lidar.ray_cast_semantic')
    sem_lidar_bp.set_attribute('rotation_frequency', str(config.lidar_rotation_frequency))
    sem_lidar_bp.set_attribute('points_per_second', str(config.lidar_points_per_second))
    sensors['semantic_lidar'] = world.spawn_actor(sem_lidar_bp, lidar_transform, attach_to=vehicle)

    return sensors


class SensorData:
    """Collects sensor data callbacks."""
    def __init__(self):
        self.rgb = None
        self.lidar = None
        self.semantic_lidar = None

    def on_rgb(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Drop alpha
        self.rgb = array

    def on_lidar(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        self.lidar = points

    def on_semantic_lidar(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
        ]))
        result = np.column_stack([
            points['x'], points['y'], points['z'], points['CosAngle'],
            points['ObjIdx'].astype(np.float64), points['ObjTag'].astype(np.float64)
        ])
        self.semantic_lidar = result

def spawn_traffic(client, world, tm, num_vehicles=30, num_walkers=20):
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_actors = []
    
    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) >= 4]

    import random
    random.shuffle(spawn_points)
    num_vehicles = min(num_vehicles, len(spawn_points) - 1)

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

    print(f"Spawned {len(vehicle_actors)} NPC vehicles (no walkers to save script complexity)")
    return vehicle_actors, []

# ============================================================
#  Main Collection Loop
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', default='Town04')
    parser.add_argument('--duration', type=int, default=3600, help='Collection duration (seconds)')
    parser.add_argument('--vehicles', type=int, default=50, help='Traffic density')
    parser.add_argument('--save-dir', default='/storage/dataset', help='Directory to save dataset')
    args = parser.parse_args()

    date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    dataset_name = f'dataset_{args.town}_{date_str}'
    base_dir = os.path.join(args.save_dir, dataset_name, 'route_00')
    
    for subdir in ['rgb', 'lidar', 'measurements', 'radar_data_front_86', 'radar_data_rear_86', 'boxes']:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

    actors = []
    npc_vehicle_ids, walker_ids = [], []

    try:
        print("Connecting to CARLA...")
        client, world, tm = setup_carla(args.host, args.port, args.town)
        print(f"Connected to CARLA!")

        vehicle = spawn_vehicle(world)
        actors.append(vehicle)

        vehicle.set_autopilot(True, tm.get_port())
        tm.distance_to_leading_vehicle(vehicle, 3.0)
        tm.vehicle_percentage_speed_difference(vehicle, -10.0) # slightly faster

        sensor_data = SensorData()
        config = SimpleConfig()
        sensors = attach_sensors(world, vehicle, config)
        sensors['camera'].listen(sensor_data.on_rgb)
        sensors['lidar'].listen(sensor_data.on_lidar)
        sensors['semantic_lidar'].listen(sensor_data.on_semantic_lidar)
        actors.extend(sensors.values())

        spectator = world.get_spectator()
        for _ in range(10): world.tick()

        npc_vehicle_ids, _ = spawn_traffic(client, world, tm, args.vehicles, 0)

        print("\n" + "=" * 50)
        print(f" Data Collection Started! Saving to: {base_dir}")
        print("=" * 50 + "\n")

        start_time = time.time()
        start_frame = -1
        frame_idx = 0
        
        while time.time() - start_time < args.duration:
            world.tick()

            if sensor_data.rgb is None or sensor_data.lidar is None or sensor_data.semantic_lidar is None:
                continue

            if start_frame == -1:
                start_frame = world.get_snapshot().frame

            current_frame = world.get_snapshot().frame
            
            # Save data every 10 frames (2 FPS)
            if (current_frame - start_frame) % 10 != 0:
                continue

            timestamp_str = f"{frame_idx:04d}"

            # 1. RGB
            cv2.imwrite(os.path.join(base_dir, 'rgb', f'{timestamp_str}.jpg'), sensor_data.rgb)

            # 2. LiDAR (.laz format)
            try:
                lidar_xyz = sensor_data.lidar[:, :3]
                header = laspy.LasHeader(point_format=0)
                header.offsets = np.min(lidar_xyz, axis=0) if len(lidar_xyz) > 0 else np.zeros(3)
                header.scales = np.array([0.001, 0.001, 0.001])
                
                with laspy.open(os.path.join(base_dir, 'lidar', f'{timestamp_str}.laz'), mode='w', header=header) as writer:
                    if len(lidar_xyz) > 0:
                        point_record = laspy.ScaleAwarePointRecord.zeros(lidar_xyz.shape[0], header=header)
                        point_record.x = lidar_xyz[:, 0]
                        point_record.y = lidar_xyz[:, 1]
                        point_record.z = lidar_xyz[:, 2]
                        writer.write_points(point_record)
            except Exception as e:
                print(f"Warning: Lidar saving failed - {e}")

            # 3. Radar (2D heatmap from sem lidar)
            try:
                raw_radar = sensor_data.semantic_lidar
                radar_front = convert_sem_lidar_2D_img_func(raw_radar, 0)
                radar_rear = convert_sem_lidar_2D_img_func(raw_radar, 180)
                
                radar_front = radar_front * mask_for_radar
                radar_rear = radar_rear * mask_for_radar
                
                np.save(os.path.join(base_dir, 'radar_data_front_86', timestamp_str), radar_front)
                np.save(os.path.join(base_dir, 'radar_data_rear_86', timestamp_str), radar_rear)
            except Exception as e:
                print(f"Warning: Radar saving failed - {e}")

            # 4. Measurements
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            control = vehicle.get_control()
            
            meas = {
                "steer": control.steer,
                "throttle": control.throttle,
                "brake": control.brake > 0.5,
                "speed": speed,
                "target_speed": 5.0 if speed > 2.0 else 0.0,
                "angle": control.steer * 90.0,
                "theta": math.radians(transform.rotation.yaw),
                "command": 4, # 4 = FOLLOW_LANE
                "next_command": 4,
                "light_hazard": False,
                "stop_sign_hazard": False,
                "junction": False,
                "route": [[transform.location.x, transform.location.y, transform.location.z]]
            }
            
            # Save dummy empty boxes for compatibility
            with gzip.open(os.path.join(base_dir, 'boxes', f'{timestamp_str}.json.gz'), 'wt', encoding='utf-8') as f:
                json.dump([], f, indent=4)
                
            with gzip.open(os.path.join(base_dir, 'measurements', f'{timestamp_str}.json.gz'), 'wt', encoding='utf-8') as f:
                json.dump(meas, f, indent=4)

            print(f"[{frame_idx:04d}] Saved! Speed: {speed:.1f} m/s | Steer: {control.steer:.2f} | Throttle: {control.throttle:.2f}")
            frame_idx += 1

            # Update camera
            loc = transform.transform(carla.Location(x=-6.0, z=3.0))
            spectator.set_transform(carla.Transform(loc, carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw)))

    except KeyboardInterrupt:
        print("\nStopping data collector...")
    finally:
        print("Cleaning up...")
        if 'client' in locals():
            try:
                client.apply_batch([carla.command.DestroyActor(x) for x in npc_vehicle_ids])
            except: pass
        for actor in reversed(actors):
            try: actor.destroy()
            except: pass
            
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except: pass
        print("Done!")

if __name__ == '__main__':
    main()
