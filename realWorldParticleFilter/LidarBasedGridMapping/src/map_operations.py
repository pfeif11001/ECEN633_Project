import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from occupancy import OccupancyGrid
from plot_operations import plot_map
import time

def normalize_angle(angle):
    """
    Normalize an angle to be within [-π, π].
    :param angle: Angle in radians.
    :return: Normalized angle.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def initialise_map(config, odometry):
    """
    Initialises the map based on the given configuration and odometry data.
    
    :param config: The configuration dictionary.
    :param odometry: The odometry data array.
    :return: An initialised OccupancyGrid object.
    """
    probability = config["map"]["prob_occ"]
    map_x_size, map_y_size = config['map']['size']
    resolution = config['map']['resolution']  # Map resolution multiplier from configuration
    
    # Adjust odometry start position to the center of x-axis and 1/3 of y-axis
    odometry[:, 0] += map_x_size / 2
    odometry[:, 1] += map_y_size / 3
    
    map_size = [map_x_size, map_y_size]
    return OccupancyGrid(map_size, config)

def process_odometry_and_laser_data(config, odometry, laser, map):
    """
    Processes odometry and laser data and updates the map.
    
    :param config: The configuration dictionary.
    :param odometry: The odometry data array.
    :param laser: The laser data array.
    :param map: The OccupancyGrid object to be updated.
    """
    plt.figure(1)
    plt.ion()
    
    laser_range = config["laser"]["max_range"]
    laser_clipped = np.clip(laser, 0, laser_range)  # Limit the sensor readings to the max range

    data = {'X_t': [], 'U_t': [], 'Z_tp1': []}
    
    # Loop through each odometry and laser data point to update the map
    for i in tqdm(range(len(odometry)), desc="Processing data"):
        # Update the map with current odometry and laser data
        map.update(odometry[i], laser_clipped[i])
        plot_map(config, odometry, laser_clipped, map, i)
        plt.pause(0.01)  # Ensure the plot updates
        
        # Copy odometry and convert angle to degrees for logging
        data['X_t'].append(odometry[i])
        
        # Calculate control input (Δx, Δy, Δθ)
        if i > 0:
            control = odometry[i] - odometry[i - 1]
            control[2] = normalize_angle(control[2])  # Normalize angle difference
        else:
            # control = [odometry[0,0], odometry[0, 1], odometry[0, 2]]
            control = [0, 0, 0]

        
        data['U_t'].append(control)
        data['Z_tp1'].append(laser_clipped[i])
    
    # Generate angles for the laser data
    data['angles'] = np.linspace(-90, 90, len(laser[0]))
    
    # Save data to a uniquely named .npz file
    np.savez(f"data.npz", **data)
