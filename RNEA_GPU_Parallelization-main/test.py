import numpy as np
from Pgenerate import Pgenerate
import Ygenerate
from timeit import default_timer as timer
from URDFParser import URDFParser
from URDFParser import Robot
from util import parseInputs, printUsage, validateRobot, initializeValues, printErr
from RBDReference import RBDReference
import time
# 初始化 URDF 解析器和机器人
parser = URDFParser()
robot = parser.parse(r'E:\RNEA_GPU_Parallelization-main\URDFParser\iiwa.urdf')

# 验证机器人模型是否正确
validateRobot(robot)
Imat_arr = []
for ind in range(7):
    Imat_arr.append(robot.get_Imat_by_id(ind))
Imat_arr = np.array(Imat_arr)
inertial_parameters = Pgenerate(Imat_arr)

q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
q_dot = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
qr_dot = np.array([0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72])
qr_ddot = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])

start_time = time.time()
for i in range(10000):
    # Define input array
    matrix = Ygenerate.compute_Y(q, q_dot, qr_dot, qr_ddot)
    torque = np.dot(matrix, inertial_parameters)

end_time = time.time()
print(end_time - start_time)

# # 显示结果
# print("Matrix Y:")
# print(matrix)
# print("Y 的形状:", matrix.shape)
# print("torque")
# print(np.dot(matrix,inertial_parameters))

