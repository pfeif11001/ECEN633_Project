import numpy as np


# Load in localization-dataset.npz
data = np.load("localization-dataset.npz")
X_t = data['X_t']       # X_t is the true position of the robot at time t
U_t = data['U_t']       # U_t is the control input at time t
Z_tp1 = data['Z_tp1']   # Z_tp1 is the observation at time t+1
angles = data['angles'] # angles is the angles of the LIDAR

