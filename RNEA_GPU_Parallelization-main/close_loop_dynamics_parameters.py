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
import pandas as pd

# 初始化 URDF 解析器和机器人
parser = URDFParser()
robot = parser.parse(r'E:\RNEA_GPU_Parallelization-main\URDFParser\lower_arm.urdf')

# 验证机器人模型是否正确
validateRobot(robot)

# 初始化机器人状态
reference = RBDReference(robot)
q, qd, u, n = initializeValues(robot, MATCH_CPP_RANDOM=True)

# 定义关节状态
q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
qd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
qdd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

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

np.set_printoptions(precision=6)
parent_id_arr = np.array(parent_id_arr)
S_arr = np.array(S_arr)
Imat_arr = np.array(Imat_arr)
print(Imat_arr)



#write xmat functions to file
import array
import sys
import os
import inspect

for ind in range(n):
  with open(f'./xmat_lower/xmat{ind}.py', 'w') as f:
      original_stdout = sys.stdout
      sys.stdout = f
      try:
          print("from numpy import array, sin, cos")
          print()
          content = robot.get_Xmat_Func_by_id(ind)
          source_code = inspect.getsource(content)
          print(source_code)
          # print(content)
      finally:
          sys.stdout = original_stdout
          f.close()
# 定义必要的函数
def cross_operator(d_vec):
    d_output = np.zeros((6, 6))
    d_output[0, 1] = -d_vec[2]
    d_output[0, 2] = d_vec[1]
    d_output[1, 0] = d_vec[2]
    d_output[1, 2] = -d_vec[0]
    d_output[2, 0] = -d_vec[1]
    d_output[2, 1] = d_vec[0]

    d_output[3, 1] = -d_vec[5]
    d_output[3, 2] = d_vec[4]
    d_output[3, 4] = -d_vec[2]
    d_output[3, 5] = d_vec[1]
    d_output[4, 0] = d_vec[5]
    d_output[4, 2] = -d_vec[3]
    d_output[4, 3] = d_vec[2]
    d_output[4, 5] = -d_vec[0]
    d_output[5, 0] = -d_vec[4]
    d_output[5, 1] = d_vec[3]
    d_output[5, 3] = -d_vec[1]
    d_output[5, 4] = d_vec[0]
    return d_output

def mxS_numpy(S, vec, alpha=1):
    vec_output = cross_operator(vec)
    mxS_output = alpha * np.dot(vec_output, S)
    return mxS_output

def vxIv_numpy(vec, Imat):
    temp = np.dot(Imat, vec)
    vecXIvec = np.zeros(6)
    vecXIvec[0] = -vec[2]*temp[1] + vec[1]*temp[2] - vec[5]*temp[4] + vec[4]*temp[5]
    vecXIvec[1] = vec[2]*temp[0] - vec[0]*temp[2] + vec[5]*temp[3] - vec[3]*temp[5]
    vecXIvec[2] = -vec[1]*temp[0] + vec[0]*temp[1] - vec[4]*temp[3] + vec[3]*temp[4]
    vecXIvec[3] = -vec[2]*temp[4] + vec[1]*temp[5]
    vecXIvec[4] = vec[2]*temp[3] - vec[0]*temp[5]
    vecXIvec[5] = -vec[1]*temp[3] + vec[0]*temp[4]
    return vecXIvec

def rnea_fpass_numpy(q, qd, qdd, GRAVITY=0):
    n = len(q)
    v = np.zeros((6, n))
    a = np.zeros((6, n))
    f = np.zeros((6, n))

    gravity_vec = np.zeros(6)
    gravity_vec[5] = -GRAVITY

    for ind in range(n):
        parent_ind = parent_id_arr[ind]
        # 获取Xmat函数并计算Xmat
        Xmat_func = robot.get_Xmat_Func_by_id(ind)
        Xmat = Xmat_func(q[ind])
        S = S_arr[ind]

        if parent_ind == -1:
            a[:, ind] = np.dot(Xmat, gravity_vec)
            v[:, ind] = S * qd[ind]
            a[:, ind] += mxS_numpy(S, v[:, ind], qd[ind]) + S * qdd[ind]
        else:
            v[:, ind] = np.dot(Xmat, v[:, parent_ind]) + S * qd[ind]
            a[:, ind] = np.dot(Xmat, a[:, parent_ind]) + mxS_numpy(S, v[:, ind], qd[ind]) + S * qdd[ind]

        Imat = Imat_arr[ind]
        f[:, ind] = np.dot(Imat, a[:, ind]) + vxIv_numpy(v[:, ind], Imat)

    return v, a, f


def rnea_bpass_numpy(f):
    n = len(q)
    c = np.zeros(n)

    for ind in range(n - 1, -1, -1):
        S = S_arr[ind]
        c[ind] = np.dot(S, f[:, ind])

        parent_ind = parent_id_arr[ind]
        if parent_ind != -1:
            # 获取Xmat函数并计算Xmat
            Xmat_func = robot.get_Xmat_Func_by_id(ind)
            Xmat = Xmat_func(q[ind])
            f[:, parent_ind] += np.dot(Xmat.T, f[:, ind])

    return c

def rnea_numpy(q, qd, qdd, GRAVITY=-9.8):
    v, a, f = rnea_fpass_numpy(q, qd, qdd, GRAVITY)
    c = rnea_bpass_numpy(f)
    return c
# 设置显示的小数位数，例如保留2位小数
np.set_printoptions(precision=6)
# 执行逆动力学计算
start_time = timer()
tau = rnea_numpy(q, qd, qdd,0)
np.set_printoptions(precision=8)

end_time = timer()
print("计算得到的关节力矩tau:", tau)
print("计算时间:", end_time - start_time, "秒")
