import numpy as np
import imageio
import pandas as pd


loaded_arr = np.load('map.npy')

# Threshold the image
for i in range(loaded_arr.shape[0]):
    for j in range(loaded_arr.shape[1]):
        if loaded_arr[i, j] > 0.5:
            loaded_arr[i, j] = 0
        else:
            loaded_arr[i, j] = 1

# Convert the array to integers
loaded_arr = loaded_arr.astype(int)


df = pd.DataFrame(loaded_arr)

# Save the map as a csv file
# np.savetxt('map.csv', loaded_arr, delimiter=',')
df.to_csv('real_world_map.csv', index=False, header=False)

# Convert the array to uint8
loaded_arr_uint8 = (loaded_arr * 255 / np.max(loaded_arr)).astype(np.uint8)

# Save numpy array as a png image
imageio.imwrite('map.png', loaded_arr_uint8)