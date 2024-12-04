#!/usr/bin/env python3
"""Module defining an implementation of particle filter localization.

Defined Classes: 
ParticleFilterLocalizer - Implements particle filter localization.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import math
import pickle
import argparse

from numba import njit
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from lab3.occupancy_grid_map import OccupancyGrid 

__author__ = "Joshua Mangelson"
__copyright__ = "Copyright 2020, Joshua Mangelson, Brigham Young University"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"

# Helper functions
def convert_xyt_to_H(dx:float, dy:float, dtheta:float) -> 'np.ndarray[(3,3) , np.dtype[np.float64]]':
    """Convert X-Y-Theta to a transformation matrix.

    Parameters: 
    dx: Change in x. 
    dy: Change in y.
    dtheta: Change in orientation (In degrees).
    
    Return: 
    H: A 2D homogeous transformation matrix encoding the input parameters.
    """
    dtheta = math.radians(dtheta)

    H = np.array(
        [ [ math.cos(dtheta), -math.sin(dtheta), dx],
          [ math.sin(dtheta), math.cos(dtheta), dy],
          [ 0, 0, 1] ] )
    return H

@njit
def get_cell_odds(grid: 'np.ndarray[(num_grid_squares_x, num_grid_squares_y) , np.dtype[np.float64]]', x: float, y: float,
                  xmin: float, ymin: float, res: float) -> float:
    """Get cell odds of a specific x,y coordinate. Note that this
    function DOES NOT check for out of bounds x, y coordinates.
    
    Parameters:
    grid: 2D numpy array of grid probabilities, found in OccupancyGrid.grid
    x: x coordinate (not an index)
    y: y coordinate (not an index)
    xmin: Minimum x-coordinate of the grid. Found in OccupancyGrid.xlim[0]
    ymin: Minimum y-coordinate of the grid. Found in OccupancyGrid.ylim[0]
    res: Resolution of the grid. Found in OccupancyGrid.resolution

    Returns:
    Log-prob of that x-y coordinate being occupied.
    
    """
    x_idx = int((x - xmin) / res)
    y_idx = int((y - ymin) / res)
    
    return grid[y_idx, x_idx]

    
class ParticleFilterLocalizer():
    """Class that utilizes a particle filter to localize in a prior ogm."""

    def __init__(self, priormap: OccupancyGrid, xlim: float, ylim: float, N: int,
                 alphas: 'np.ndarray[(4,) , np.dtype[np.float64]]', cov: 'np.ndarray[(3,) , np.dtype[np.float64]]'):
        """Create and initialize the particle filter.

        Parameters:
        priormap: An OccupancyGrid object representing the underlying map.
        xlim: A two element list defining the x bounds to be searched.
        ylim: A two element list defining the y bounds to be searched.
        N: The number of particles to generate
        alphas: A 4-array of values make up the laser sensor model. In the order of [p_hit, p_unexp, p_random, p_max]. Should sum to 1.
        cov: A 3-array of cov for the motion model. In the order of [x, y, theta]
        """
        self.priormap = priormap
        self.N = N
        self.weights = np.ones(N) / N
        
        #set up particles
        self.particles = np.random.uniform([xlim[0], ylim[0], 0],
                                        [xlim[1], ylim[1], 360], (N,3))
        # We sample one particle exactly to jumpstart the algorithm
        # self.particles[0] = np.array([1.5, 1.5, 0])
        self.particles[0] = np.array([50.255856, 33.34968063, 35.45938391])

        # Set up noise for motion/measurement models
        # You may do whatever calculations you want, but 
        # make sure you use what's passed into __init__()
        self.cov = cov
        self.alphas = alphas
        self.z_max = 20
    
    
    def propagate_motion(self, u: 'np.ndarray[(3,), np.dtype[np.float64]]'):
        """Propagate motion noisily through all particles in place.

        Make sure you use self.cov somewhere for the noise added.

        Parameters:
        u: A 3-array representing movement from the previous step 
        as a delta x, delta y, delta theta (in degrees).
        """
        dx, dy, dtheta = u
        dtheta_rad = math.radians(dtheta)  # Convert dtheta to radians.

        for i in range(self.N):
            # Add noise to the deltas.
            dx_noisy = dx + np.random.normal(0, self.cov[0])
            dy_noisy = dy + np.random.normal(0, self.cov[1])
            dtheta_noisy = dtheta_rad + np.random.normal(0, self.cov[2])

            # Get the current state of the particle.
            x, y, theta = self.particles[i]
            theta_rad = math.radians(theta)

            # Compute the updated position directly.
            new_x = x + dx_noisy
            new_y = y + dy_noisy
            new_theta = (theta_rad + dtheta_noisy) % (2 * math.pi)

            # Update the particle with the new state.
            self.particles[i, 0] = new_x
            self.particles[i, 1] = new_y
            self.particles[i, 2] = math.degrees(new_theta)  # Store theta in degrees as in the original.


    @staticmethod
    @njit
    def expected_measurement(angles: 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]',
                             pos: 'np.ndarray[(3,) , np.dtype[np.float64]]',
                             grid: 'np.ndarray[(num_grid_squares_x,num_grid_squares_y) , np.dtype[np.float64]]',
                             xmin: float, xmax: float, ymin: float, ymax: float,
                             res: float) -> 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]':
        """Get the expected distances of a laser range finder based on it's
            laser angles and robot position

        NOTE: To use @njit (please do it'll make it tons faster), use 
            get_cell_odds to get the odds of an x,y coordinate. Note get_cell_odds
            does NOT check for out of bounds, you'll have to do that if you use it.
            You'll also have to only use other functions that are also @njit, or are
            in the numpy library.

        If you don't want to use @njit, remove @njit from a couple of lines up.

        Note there is a few ways of doing this, the tests are flexible and accept a given range.
        Also, search out to 25 meters, 5 meters beyond the max range of sensor.

        Parameters:
        angles: Angles the LiDAR was sampled at, with respect to the local x-axis of 
            the robot (straight ahead) with positive angles laying to the left 
            of the x-axis following the right hand rule (counter clock-wise in
            the xy plane). (In Degrees).      
        pos: The position of the particle to sample the distribution of.
        grid: The 2D numpy array as stored in (get from self.priormap.grid)
        xmin: Minimum of the map in the x-direction
        xmax: Maximum of the map in the x-direction
        ymin: Minimum of the map in the y-direction
        ymax: Maximum of the map in the y-direction
        res: Resolution of the map.

        Returns:
        Numpy array of expected distances
        """
        ####################################
        # Finish this implemenation!
        
        expected = np.zeros(angles.shape)
        
        for i, angle in enumerate(angles):
            x, y, theta = pos
            theta = math.radians(theta + angle)
            
            distance = 0
            while distance <= 25:
                x += res * math.cos(theta)
                y += res * math.sin(theta)
                distance += res
                
                if x < xmin or x > xmax or y < ymin or y > ymax:
                    break
                
                if get_cell_odds(grid, x, y, xmin, ymin, res) >= 0:
                    break
            
            expected[i] = distance
            
        return expected


        ####################################

    @staticmethod
    @njit
    def update_weight(z_k: 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]',
                      z_t: 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]',
                      alphas: 'np.ndarray[(4,) , np.dtype[np.float64]]'):
        """Update a single particle's probability according to expected distances.

        NOTE: To use @njit (please do it'll make it tons faster), use 
            get_cell_odds to get the odds of an x,y coordinate. Note get_cell_odds
            does NOT check for out of bounds, you'll have to do that if you use it.
            You'll also have to only use other functions that are also @njit, or are
            in the numpy library.

        If you don't want to use @njit, remove @njit from a couple of lines up.

        Parameters:
        z_k: An array of expected measurements given robot position
        z_t: An array of range measurements from the LiDAR
        alphas: The various weights of the different probability distributions in the order
            p_hit, p_random, p_max (get from self.alphas)
        """
        ####################################
        # Finish this implemenation!!
        
        sigma_hit = 1.0
        z_max = 20.0

        weight = 1.0
        for i in range(len(z_k)):
            z_ki = z_k[i]
            z_ti = z_t[i]

            # p_hit
            p_hit = (1 / (np.sqrt(2 * np.pi * sigma_hit))) * np.exp(-0.5 * ((z_ti - z_ki) ** 2) / (sigma_hit))

            # p_unexp
            p_unexp = 0.0

            # p_max
            p_max = 1.0 if z_ti == z_max else 0.0

            # p_rand
            p_rand = 1.0 / z_max if 0 <= z_ti <= z_max else 0.0

            # Combine weights
            p = alphas[0] * p_hit + alphas[1] * p_unexp + alphas[2] * p_max + alphas[3] * p_rand

            weight *= p

        return weight
        
        

        ####################################

    def normalize_weights(self):
        """Normalize self.weights in place."""
        ####################################
        # Finish this implemenation!!
        
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        else:
            num_particles = len(self.weights)
            self.weights = [1.0 / num_particles] * num_particles
        
        

        ####################################

    def resample_particles(self):
        """Resample particles in place according to the probabilities in self.weights"""
        ####################################
        # Finish this implemenation!!
        
        cumulative_sum = np.cumsum(self.weights)

        # Generate a random starting point
        start = np.random.uniform(0, 1 / self.N)
        points = start + np.arange(self.N) / self.N

        # Create an array to hold the new particles
        new_particles = np.zeros_like(self.particles)

        # Resample particles
        j = 0
        for i in range(self.N):
            while points[i] > cumulative_sum[j]:
                j += 1
            new_particles[i] = self.particles[j]

        # Update particles with the resampled ones
        self.particles = new_particles

        ####################################

    def iterate(self, u: 'np.ndarray[(3,) , np.dtype[np.float64]]',
                z_t: 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]',
                angles: 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]'):
        """Propagate motion according to control, and correct using laser measurements.
        
        Parameters:
        u: A 3-array representing movement from the previous step 
            as a delta x, delta y, delta theta. 
        z_t: An array of range measurements from the LiDAR
        angles: Angles the LiDAR was sampled at, with respect to the local x-axis of 
            the robot (straight ahead) with positive angles laying to the left 
            of the x-axis following the right hand rule (counter clock-wise in
            the xy plane). (In Degrees).
        """
        ####################################
        # Finish this implemenation!!
        
        # Propagate motion
        self.propagate_motion(u)
        
        # Compute expected measurements for each particle
        expected_measurements = np.zeros((self.N, len(angles)))
        for i in range(self.N):
            expected_measurements[i] = self.expected_measurement(
                angles, self.particles[i], self.priormap.grid,
                self.priormap.xlim[0], self.priormap.xlim[1],
                self.priormap.ylim[0], self.priormap.ylim[1],
                self.priormap.resolution
            )
        
        # Update weights based on actual measurements
        for i in range(self.N):
            self.weights[i] = self.update_weight(expected_measurements[i], z_t, self.alphas)
        
        # Normalize weights
        self.normalize_weights()
        
        # Resample particles
        self.resample_particles()

        
        ####################################


def main(plot_live: bool, mapfile: str, datafile: str, num: int, makeGif: bool):
    
    # Import for making Gif
    gifFrames = []
    if makeGif:
        from PIL import Image

    np.random.seed(0)
        
    #################################################
    # Tweak these like you want
    alphas = np.array([0.8, 0.0, 0.05, 0.15]) # A 4-array of values make up the laser sensor model. In the order of [p_hit, p_unexp, p_random, p_max]. Should sum to 1.
    # cov = np.array([.04, .04, .01])
    cov = np.array([0.1, 0.1, 0.05])      # This one works really well!    
    #################################################

    # Load prior map
    prior_map = pickle.load(open(mapfile, "rb"))

    # Load data stream
    data = np.load(datafile)
    
    X_t = data['X_t']
    # Convert theta in X_t from radians to degrees
    X_t[:,2] = np.degrees(X_t[:,2])
    
    U_t = data['U_t']
    # Convert theta in U_t from radians to degrees
    U_t[:,2] = np.degrees(U_t[:,2])
    
    Z_tp1 = data['Z_tp1']
    
    angles = data['angles']
        
    # Initialize particle filter
    pf = ParticleFilterLocalizer(prior_map, [0, 100], [0, 100], num, alphas, cov)    

    # Setup plotting
    if plot_live:
        plt.ion()
        fig, ax = plt.subplots()    

        prior_map.plot_grid(ax)
        true_loc = ax.scatter(X_t[0][0], X_t[0][1], 2, 'b')
        particles = ax.scatter(pf.particles[:,0], pf.particles[:,1], 0.5, 'r')   
             

    # Loop through data stream
    for t in tqdm(range(len(X_t)-1)):
    # for t in tqdm(range(20)):
        # Extract data
        u_t = U_t[t]
        z_tp1 = Z_tp1[t]
        
        #######################################################
        # Operate on Data to run the particle filter algorithm
        pf.iterate(u_t, z_tp1, angles)
        
        #######################################################

        # Plot
        # if plot_live and t % 5 == 0:
        if plot_live:

            true_loc.set_offsets(X_t[t+1:t+2,:2])
            particles.set_offsets(pf.particles[:,:2])

            fig.canvas.draw()
    
            if (makeGif):
                imgData = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                mod = np.sqrt(imgData.shape[0]/(3*w*h)) # multi-sampling of pixels on high-res displays does weird things
                im = imgData.reshape((int(h*mod), int(w*mod), -1))
                gifFrames.append(Image.fromarray(im))

            fig.canvas.flush_events()
            
    if (plot_live and makeGif):
        gifFrames[0].save(
            'gifOutput.gif', 
            format='GIF',
            append_images=gifFrames[1:],
            save_all=True,
            duration=len(gifFrames)*2*0.1,
            loop=0)
        

    if plot_live:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occupancy Grid Map")
    parser.add_argument("-g", "--makeGif", action="store_true", help="Whether to save a gif of all the frames")
    parser.add_argument("-p", "--plot_live", action="store_true", help="Whether we should plot as we go")
    parser.add_argument("-d", "--datafile", type=str, default="data/localization-dataset.npz", help="Location of localization data. Defaults to data/localization-data.npz")
    parser.add_argument("-m", "--mapfile", type=str, default="data/map.p", help="Location of map data. Defaults to data/map.p")
    parser.add_argument("-n", "--num", type=int, default=100, help="Number of particles to use")
    args = vars(parser.parse_args())

    main(**args)
