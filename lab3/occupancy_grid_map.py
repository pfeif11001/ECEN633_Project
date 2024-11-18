#!/usr/bin/env python3
"""Module defining necessary objects for maintaining an ocupancy grid map.

Defined Classes:
OccupancyGrid - Class representing a generic occupancy grid. 
OccupancyGridMap - Class representing an occupancy grid map. 
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
import csv
import pickle

__author__ = "Joshua Mangelson"
__copyright__ = "Copyright 2020, Joshua Mangelson, Brigham Young University"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"

class OccupancyGrid():
    """A class defining an occupancy grid structure."""

    def __init__(self, resolution, xlim, ylim, prior):
        """Create an occupancy grid object.

        Parameters: 
        resolution: A number defining the width of cells in the grid.
        xlim: A two element list defining the x bounds of the grid. 
        ylim: A two element list defining the y bounds of the grid.

        Exceptions:
        ValueError("Resolution does not exactly subdivide xlim."): This 
          function raises a ValueError with the above message if the specified
          resolution does not exactly subdivide the range of x values 
          specified by xlim.
        ValueError("Resolution does not exactly subdivide ylim."): This 
          function raises a ValueError with the above message if the specified
          resolution does not exactly subdivide the range of y values 
          specified by ylim.
        """
        xwidth = xlim[1] - xlim[0]
        xdim = xwidth / resolution
        if(not xdim % 1 == 0):
            raise ValueError("Resolution does not exactly subdivide xlim.")
        ywidth = ylim[1] - ylim[0]
        ydim = ywidth / resolution
        if(not ydim % 1 == 0):
            raise ValueError("Resolution does not exactly subdivide ylim.")

        self.resolution = resolution
        self.xlim = xlim
        self.ylim = ylim
        self.xdim = int(xdim)
        self.ydim = int(ydim)

        self.xcoords = np.arange(xlim[0], xlim[1], resolution)
        self.ycoords = np.arange(ylim[0], ylim[1], resolution)

        self.prior = prior
        self.grid = np.full((self.ydim, self.xdim), np.log(self.prior/(1-self.prior)))

        self.plot = None

    def get_cell_index(self, x_pos, y_pos):
        """Return the cell indices for the specified position.

        Parameters:
        x_pos: The x position coordinate to be queried. (Can be a float or a ndarray)
        y_pos: The y position coordinate to be queried. (Can be a float or a ndarray)

        Exceptions:
        ValueError("X Position is outside grid."): This function raises a
          ValueError with the above message if the specified x coord. does
          not fall with in the bounds of the grid.
        ValueError("Y Position is outside grid."): This function raises a
          ValueError with the above message if the specified y coord. does
          not fall with in the bounds of the grid.

        Return:
        index: If receives a float: A 2-element list containing the indices for 
            the cell in the grid (x-index first, followed by y-index).
            If receives an array: A Nx2 array containing indices for cells in the grid.
        """
        # if it's an array
        if type(x_pos) is np.ndarray:
            if np.any((x_pos < self.xlim[0]) + (x_pos >= self.xlim[1])):
                raise ValueError("X Position is outside grid.")
            if np.any((y_pos < self.ylim[0]) + (y_pos >= self.ylim[1])):
                raise ValueError("Y Position is outside grid.")

            x = ((x_pos - self.xlim[0]) / self.resolution).astype('int')
            y = ((y_pos - self.ylim[0]) / self.resolution).astype('int')
            return np.array([x,y]).T

        # if it's a float
        else:
            if(x_pos < self.xlim[0] or x_pos >= self.xlim[1]):
                raise ValueError("X Position is outside grid.")
            if(y_pos < self.ylim[0] or y_pos >= self.ylim[1]):
                raise ValueError("Y Position is outside grid.")

            x = int((x_pos - self.xlim[0]) / self.resolution)
            y = int((y_pos - self.ylim[0]) / self.resolution)
            return [x,y]

    def get_cell_center(self, x_idx, y_idx):
        """Return the center of the cell with the given index.

        Parameters: 
        x_idx: The x index of a coordinate in the map
        y_idx: The y index of a coordinate in the map

        Exceptions:
        ValueError("X index invalid."): This function raises a
          ValueError with the above message if the specified x index does
          not fall with in the bounds of the grid.
        ValueError("Y index invalid."): This function raises a
          ValueError with the above message if the specified y index does
          not fall with in the bounds of the grid.

        Return:
        cell_center: A 2-element list containing the x and then y coordinates 
          of the center of the cell.
        """
        if(x_idx < 0 or x_idx >= self.xdim):
            raise ValueError("X index invalid.")
        if(y_idx < 0 or y_idx >= self.ydim):
            raise ValueError("Y index invalid.")        

        x = self.xcoords[x_idx] + self.resolution/2
        y = self.ycoords[y_idx] + self.resolution/2

        cell_center = [x, y]
        return cell_center
 
    def get_cell_odds(self, index):
        return self.grid[index[1]][index[0]]

    @staticmethod
    def p_from_l(l):
        try:
            ans =  1 - 1/(1 + np.exp(l))
            return ans
        except OverflowError:
            ans = 1
            return ans

    def plot_grid(self, ax):
        """Plot the grid to the screen."""
        self.grid_plt = np.vectorize(self.p_from_l)(self.grid)

        if self.plot is None:
            X, Y = np.meshgrid(self.xcoords, self.ycoords)
            self.plot = ax.pcolormesh(X, Y, self.grid_plt, shading='auto', cmap='gray_r', vmin=0, vmax=1)

        else:
            self.plot.set_array(self.grid_plt.ravel())

    def update_cell_with_meas_logodds(self, x_idx, y_idx, l):
        """Update a cell by adding the log odds corresponding to a meas.

        Parameters: 
        x_idx: The x index of a coordinate in the map
        y_idx: The y index of a coordinate in the map
        l: The log odds value to be added to the cell.
        
        Exceptions: 
        ValueError("X index invalid."): This function raises a
          ValueError with the above message if the specified x index does
          not fall with in the bounds of the grid.
        ValueError("Y index invalid."): This function raises a
          ValueError with the above message if the specified y index does
          not fall with in the bounds of the grid.
        """
        if(x_idx < 0 or x_idx >= self.xdim):
            raise ValueError("X index invalid.")
        if(y_idx < 0 or y_idx >= self.ydim):
            raise ValueError("Y index invalid.")        

        ############################################
        # Finish this implementation!!

        # Note!!!!
        # grid cells should be indexed as self.grid[y_idx, x_idx]

        self.grid[y_idx, x_idx] += l
        ##############################################

class RobotState():
    """A class defining the state of the robot."""

    def __init__(self, x, y, theta):
        """Create a robot state object.

        Parameters: 
        x: The x position of the robot with respect to the global frame. 
        y: The y position of the robot with respect to the global frame.
        theta: The angle of the robot (or forward-looking x-axis) with respect
          to the global x-axis, with positive angles laying to the left of the
          x-axis following the right hand rule (counter clock-wise in the 
          xy plane). (In Degrees).
        """

        self.x = x
        self.y = y
        self.theta = theta
    
class OccupancyGridMap():
    """A class defining an occupancy grid map object."""    

    def __init__(self, resolution, xlim, ylim, p_free, p_occup, p_prior):
        """Create an occupancy grid map object.

        Parameters: 
        resolution: A number defining the width of cells in the grid.
        xlim: A two element list defining the x bounds of the grid. 
        ylim: A two element list defining the y bounds of the grid.

        Exceptions:
        ValueError("Resolution does not exactly subdivide xlim."): This 
          function raises a ValueError with the above message if the specified
          resolution does not exactly subdivide the range of x values 
          specified by xlim.
        ValueError("Resolution does not exactly subdivide ylim."): This 
          function raises a ValueError with the above message if the specified
          resolution does not exactly subdivide the range of y values 
          specified by ylim.
        """
        self.ogrid = OccupancyGrid(resolution, xlim, ylim, p_prior)
        self.resolution = resolution

        self.p_free  = p_free
        self.p_occup = p_occup
        self.p_prior = p_prior

        self.l_occup = np.log( self.p_occup / (1 - self.p_occup) )
        self.l_free  = np.log( self.p_free  / (1 - self.p_free) )
        self.l_prior = np.log( self.ogrid.prior / (1 - self.ogrid.prior) )

    def laser_range_inverse_sensor_model(self, x_idx, y_idx, x_t, z_t):
        """Return the logodds update for a given map cell, robot state, and meas.

        You'll likely need to use self.p_free, self.p_prior, and self.p_occup here.

        Parameters:
        x_idx: The x index of a coordinate in the map
        y_idx: The y index of a coordinate in the map
        x_t: A RobotState object representing the state of the robot
          at the time the measurement was taken.
        z_t: The range returned by the laser range sensor along the 
          cooresponding ray.

        Return: 
        l: The log odds update that should be added to the cell.
        """

        ###############################
        # Finish This Implementation!!
        m_xc, m_yc = self.ogrid.get_cell_center(x_idx, y_idx)
        r = np.sqrt( (m_xc - x_t.x)**2 + (m_yc - x_t.y)**2 )

        if r < z_t - self.ogrid.resolution/2:
          return self.l_free
        elif r < z_t + self.ogrid.resolution/2:
          return self.l_occup
        else:
          return self.l_prior

        #############################

    def find_cells_to_update_for_ray(self, x_t, z_theta_t, max_range):
        """Find the set of cells that lie along a ray.

        Parameters: 
        x_t: A RobotState object representing the state of the robot at the
          time the measurement was taken. 
        z_theta_t: The angle of the ray with respect to the local x-axis of 
          the robot (straight ahead) with positive angles laying to the left 
          of the x-axis following the right hand rule (counter clock-wise in
          the xy plane). (In Degrees).
        max_range: The maximum range to be updated by a measurement.

        Return: 
        cell_index_list: A list of cell index pairs that lists each cell that
          should be updated for a given range measurement. Each cell should 
          be listed only once. 
        """

        ####################################
        # Finish this implemenation!!
        z_possible = np.arange(0, max_range+self.ogrid.resolution/2, self.ogrid.resolution/2)
        theta = x_t.theta + z_theta_t

        # go along beam and find all cells that lie in line of sight
        x = z_possible * np.cos(theta * np.pi/180) + x_t.x
        y = z_possible * np.sin(theta * np.pi/180)  + x_t.y

        # Mask out ones outside of the map
        x_mask = np.logical_and(self.ogrid.xlim[0] <= x, x < self.ogrid.xlim[1])
        y_mask = np.logical_and(self.ogrid.ylim[0] <= y, y < self.ogrid.ylim[1])
        mask = np.logical_and(x_mask, y_mask)
        
        cell_index_list = self.ogrid.get_cell_index(x[mask], y[mask])

        return np.unique(cell_index_list, axis=0)
        ####################################
                

    def integrate_laser_range_ray(self, x_t, z_theta_t, z_t):
        """Integrate a laser range measurement into the map. 

        Parameters: 
        x_t: A RobotState object representing the state of the robot at the
          time the measurement was taken. 
        z_theta_t: The angle of the ray with respect to the local x-axis of 
          the robot (straight ahead) with positive angles laying to the left 
          of the x-axis following the right hand rule (counter clock-wise in
          the xy plane). (In Degrees).
        z_t: The range returned by the laser range sensor along the 
          cooresponding ray.
        """

        #####################################
        # Finish this implemenation!!
        # get cells in the line of sight
        # note we don't care about things much farther after the end of our laser
        cells = self.find_cells_to_update_for_ray(x_t, z_theta_t, min(20, z_t+self.ogrid.resolution*2))
        for x_idx, y_idx in cells:
            # find odds of getting that measurement
            l = self.laser_range_inverse_sensor_model(x_idx, y_idx, x_t, z_t)

            # update model with that log odds
            self.ogrid.update_cell_with_meas_logodds(x_idx, y_idx, l-self.l_prior)
        #####################################


def main(datafile, plot_live, resolution):
    # Make map
    omap = OccupancyGrid(resolution=resolution,
                            xlim=[-20,50],
                            ylim=[-30,30],
                            prior=0.5)

    reader = csv.reader(open("data/ogm_ground_truth.csv"), delimiter=",")
    x = list(reader)
    omap.grid = np.array(x).astype("float")

    omap.grid[omap.grid == 1] = 100
    omap.grid[omap.grid == 0] = -100

    pickle.dump(omap, open('data/map.p', 'wb'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occupancy Grid Map")
    parser.add_argument("-p", "--plot_live", action="store_true", help="Whether we should plot as we go")
    parser.add_argument("-d", "--datafile", type=str, default="data/rooms-dataset.npz", help="Location of data. Defaults to data/rooms-dataset.npz")
    parser.add_argument("-r", "--resolution", type=float, default=0.5, help="Grid resolution. Default should work well.")
    args = vars(parser.parse_args())

    main(**args)