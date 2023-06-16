import numpy as np
import open3d as o3d
from fps.fps_v1 import FPS

ball4096_root = 'template/balls/4096.xyz'

ball4096 = np.loadtxt(ball4096_root)[:,:3]
fps1 = FPS(ball4096, 2048)
fps2 = FPS(ball4096, 2048)
fps3 = FPS(ball4096, 2048)
fps4 = FPS(ball4096, 2048)

ball_2048_1 = fps1.fit().get()
ball_2048_2 = fps2.fit().get()
ball_2048_3 = fps3.fit().get()
ball_2048_4 = fps4.fit().get()

np.savetxt('2048_1.xyz', ball_2048_1, fmt="%.6f")
np.savetxt('2048_2.xyz', ball_2048_2, fmt="%.6f")
np.savetxt('2048_3.xyz', ball_2048_3, fmt="%.6f")
np.savetxt('2048_4.xyz', ball_2048_4, fmt="%.6f")