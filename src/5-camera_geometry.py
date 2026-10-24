# define the intrisic para and construct 3x3 numpy matrix


import numpy as np
f_x, f_y, c_x, c_y = 517.3,516.5,318.6,255.3

k=[[f_x, 0, c_x],
   [0,f_y,c_y], 
   [0,0,1]
   ]
K = np.array(k, dtype=np.float64)

print(f"Camera intrisic matric K: {K}")

P=(2,1,5)
X=P[0]
Y=P[1]
Z=P[2]

normalized_pt = np.array([(X/Z), (Y/Z), 1])
result = K @ normalized_pt
print(f"projected pixel coord: {result}")

