import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from OccupancyGrid import OccupancyGrid
from scipy.ndimage import gaussian_filter
import math


def main():

    jsonFile = "../DataSet/PreprocessedData/intel_corrected_log"
    with open(jsonFile, 'r') as f:
        input = json.load(f)
        gtData = input['map']
        
        
    # Sort the ground truth data by timestamp
        
        
        
    count = 0

    rawMoveList, gtMoveList, rawMinusGtMoveList, turningAngleList = [], [], [], []
    errorTheta1List, errorTheta2List, errorTheta2AsDirList = [], [], []
    X_t, U_t, Z_tp1 = [], [], []

    for key in sorted(gtData.keys()):
        count += 1
        if count == 1:
            prevGtReading = gtData[key]
            continue
        gtReading = gtData[key]
        prevGtX, prevGtY, prevGtTheta, prevGtMeasure = prevGtReading['x'], prevGtReading['y'], prevGtReading['theta'], prevGtReading['range']
        gtX, gtY, gtTheta, gtMeasure = gtReading['x'], gtReading['y'], gtReading['theta'], gtReading['range']

        # Append true position (X_t)
        X_t.append((gtX, gtY))

        # Calculate control input (U_t)
        delta_x = gtX - prevGtX
        delta_y = gtY - prevGtY
        delta_theta = gtTheta - prevGtTheta
        U_t.append((delta_x, delta_y, delta_theta))

        # Append range measurements (Z_tp1)
        Z_tp1.append(gtMeasure)

        # Update previous ground truth reading
        prevGtReading = gtReading

    # Create data
    data = {}
    data['X_t'] = np.asarray(X_t)
    data['U_t'] = np.asarray(U_t)
    data['Z_tp1'] = np.asarray(Z_tp1)
    
    first_key = sorted(gtData.keys())[0]
    numPoints = len(gtData[first_key]['range'])
    angles = np.linspace(0, 360, num=numPoints)  # angles in degrees
    data['angles'] = angles
    
    gtXValues = [reading['x'] for reading in gtData.values()]
    gtYValues = [reading['y'] for reading in gtData.values()]

    maxGtX = max(gtXValues)
    minGtX = min(gtXValues)
    maxGtY = max(gtYValues)
    minGtY = min(gtYValues)

    print(f"Largest ground truth x value: {maxGtX}")
    print(f"Smallest ground truth x value: {minGtX}")
    print(f"Largest ground truth y value: {maxGtY}")
    print(f"Smallest ground truth y value: {minGtY}")
    
    # Save data as npz file
    np.savez('realWorld-dataset.npz', **data)
    
    # Print the x and y values of the ground truth data
    print("Ground truth x values:")
    print(gtXValues)
    


if __name__ == '__main__':
    main()