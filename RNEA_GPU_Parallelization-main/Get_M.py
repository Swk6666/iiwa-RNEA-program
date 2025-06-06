import numpy as np
import sys
import os
import inspect
from timeit import default_timer as timer
from URDFParser import URDFParser
from URDFParser import Robot
from util import parseInputs, printUsage, validateRobot, initializeValues, printErr
from RBDReference import RBDReference
from GRiDCodeGenerator import GRiDCodeGenerator
from xmat_files import xmat0, xmat1, xmat2, xmat3, xmat4, xmat5, xmat6
# 初始化 URDF 解析器和机器人
parser = URDFParser()
robot = parser.parse(r'E:\RNEA_GPU_Parallelization-main\URDFParser\iiwa.urdf')

# 验证机器人模型是否正确
validateRobot(robot)

# 初始化机器人状态
reference = RBDReference(robot)
q, qd, u, n = initializeValues(robot, MATCH_CPP_RANDOM=True)

# 定义关节状态
q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])*0
qd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])*0
qdd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])*0

reference = RBDReference(robot)
# 获取关节数量
n = robot.get_num_joints()


# 初始化机器人矩阵和向量
parent_id_arr = []
S_arr = []
Imat_arr = []
for ind in range(n):
    parent_id_arr.append(robot.get_parent_id(ind))
    S_arr.append(robot.get_S_by_id(ind).astype(np.float64))
    Imat_arr.append(robot.get_Imat_by_id(ind))

parent_id_arr = np.array(parent_id_arr)
S_arr = np.array(S_arr)
Imat_arr = np.array(Imat_arr)

# Define a list of the functions corresponding to xmat0 to xmat6
xmat_func_arr = [xmat0._lambdifygenerated, xmat1._lambdifygenerated, xmat2._lambdifygenerated,
                 xmat3._lambdifygenerated, xmat4._lambdifygenerated, xmat5._lambdifygenerated, xmat6._lambdifygenerated]

# Number of elements
n = 7  # or the appropriate value based on your needs
# Initialize the array to store results
Xmat_arr = np.zeros((7, 6, 6))
# Evaluate the functions and store the results
q = np.array([0,0,0,0,0,0,0])  # Example q values, replace with actual q
# Evaluate the xmat functions and store the results
for ind in range(7):
    Xmat_arr[ind:, : ] = xmat_func_arr[ind](q[ind])
# print(Imat_arr)
import numpy as np

def get_M(Ic, i_X_p, Si, n_joints, q):
    """Internal function for calculating the inertia matrix."""
    
    M = np.zeros((n_joints, n_joints))  # Initialize inertia matrix as zero
    Ic_composite = [None] * len(Ic)  # List to store composite inertia matrices
    
    # Copy initial inertia matrices into Ic_composite
    for i in range(n_joints):
        Ic_composite[i] = Ic[i]
    
    # Calculate composite inertia matrix
    for i in range(n_joints-1, -1, -1):
        if i != 0:
            Ic_composite[i-1] = Ic[i-1] + np.dot(i_X_p[i].T, np.dot(Ic_composite[i], i_X_p[i]))


    # Calculate the inertia matrix
    for i in range(n_joints):
        fh = np.dot(Ic_composite[i], Si[i])
        M[i, i] = np.dot(Si[i].T, fh)  # Diagonal elements

        j = i
        while j != 0:
            fh = np.dot(i_X_p[j].T, fh)
            print("fh",j,fh)
            j -= 1
            M[i, j] = np.dot(Si[j].T, fh)
            M[j, i] = M[i, j]  # Ensure symmetry

    return M
M = get_M(Imat_arr, Xmat_arr, S_arr, 7, q)
np.set_printoptions(precision=8, suppress=True)
print(M)