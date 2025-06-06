import numpy as np
import modern_robotics as mr
from mr_urdf_loader import loadURDF
import os
from Pgenerate import Pgenerate
from timeit import default_timer as timer
from URDFParser import URDFParser
from URDFParser import Robot
from util import parseInputs, printUsage, validateRobot, initializeValues, printErr
from RBDReference import RBDReference
import time
# 定义 package 路径的实际位置

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

package_path = "D:/项目文件/my_iiwa-master/my_iiwa-master/iiwa_description"


def resolve_package_path(urdf_file):
    # 读取 URDF 文件内容
    with open(urdf_file, 'r', encoding='utf-8') as f:
        urdf_content = f.read()

    # 替换 package:// 前缀为实际路径
    urdf_content = urdf_content.replace(
        "package://iiwa_description",
        os.path.abspath(package_path)
    )

    # 写入到一个临时文件
    temp_urdf_file = urdf_file.replace(".urdf", "_resolved.urdf")
    with open(temp_urdf_file, 'w', encoding='utf-8') as f:
        f.write(urdf_content)

    return temp_urdf_file

# 使用解析后的文件
resolved_urdf_name = resolve_package_path(r"E:\paper_add_content\adaptive_control-master_1211kaiti_bianzhiliang\iiwa_description\urdf\iiwa14.urdf")
MR = loadURDF(resolved_urdf_name)

M = MR["M"]
Slist = MR["Slist"]
Mlist = MR["Mlist"]
Glist = MR["Glist"]
Blist = MR["Blist"]
g = np.array([0, 0, 0])
Ftip = np.array([0, 0, 0, 0, 0, 0])
def circle(V):
    return np.array([
        [V[0], V[1], V[2], 0, 0, 0, 0, V[5], -V[4], 0],
        [0, V[0], 0, V[1], V[2], 0, -V[5], 0, V[3], 0],
        [0, 0, V[0], 0, V[1], V[2], V[4], -V[3], 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -V[2], V[1], V[3]],
        [0, 0, 0, 0, 0, 0, V[2], 0, -V[0], V[4]],
        [0, 0, 0, 0, 0, 0, -V[1], V[0], 0, V[5]]
    ])


def get_adaptive_identification_matrix( qs, dqs, dqrs, ddqrs):
    print("get_adaptive_identification")
    Mi = np.eye(4)
    Ai = np.zeros((6, 7))
    AdTi = [[None]] * (7 + 1)
    Vi = np.zeros((6, 7 + 1))
    Vri = np.zeros((6, 7 + 1))
    Vdri = np.zeros((6, 7 + 1))
    Vdri[:, 0] = np.r_[[0, 0, 0], -g]
    AdTi[7] = mr.Adjoint(mr.TransInv(Mlist[7]))
    for i in range(7):
        Mi = np.dot(Mi, Mlist[i])
        Ai[:, i] = np.dot(mr.Adjoint(mr.TransInv(Mi)), Slist[:, i])
        AdTi[i] = mr.Adjoint(np.dot(mr.MatrixExp6(mr.VecTose3(Ai[:, i] * -qs[i])), mr.TransInv(Mlist[i])))
        Vi[:, i + 1] = np.dot(AdTi[i], Vi[:, i]) + Ai[:, i] * dqs[i]
        Vri[:, i + 1] = np.dot(AdTi[i], Vri[:, i]) + Ai[:, i] * dqrs[i]
        Vdri[:, i + 1] = np.dot(AdTi[i], Vdri[:, i]) + Ai[:, i] * ddqrs[i] + np.dot(mr.ad(Vi[:, i + 1]), Ai[:, i]) * \
                         dqrs[i]

    Yi = np.zeros((6, 10 * 7))
    Y = np.zeros((7, 10 * 7))
    for i in range(7 - 1, -1, -1):
        # AdTi给出连杆坐标系i-1变换到连杆坐标系i的伴随变换矩阵，使得运动旋量能够在这两个坐标系下转换
        Yi = AdTi[i + 1].T @ Yi  # 这里加了转置，那就是将连杆7的
        Yi[:, i * 10: (i + 1) * 10] = circle(Vdri[:, i + 1]) \
                                                                          - 0.5 * mr.ad(Vi[:, i + 1]).T @ circle(
            Vri[:, i + 1]) \
                                                                          - 0.5 * mr.ad(Vri[:, i + 1]).T @ circle(
            Vi[:, i + 1]) \
                                                                          + 0.5 * circle(
            mr.ad(Vri[:, i + 1]) @ Vi[:, i + 1])
        Y[i, :] = Ai[:, i].T @ Yi
    return Y

def inv_dynamics_adaptive(qs, dqs, dqrs, ddqrs):
    Mi = np.eye(4)
    Ai = np.zeros((6, 7))
    AdTi = [[None]] * (7 + 1)
    Vi = np.zeros((6, 7 + 1))
    Vri = np.zeros((6, 7 + 1))
    Vdri = np.zeros((6, 7 + 1))
    Vdri[:, 0] = np.r_[[0, 0, 0], -g]
    AdTi[7] = mr.Adjoint(mr.TransInv(Mlist[7]))
    Fi = np.zeros(6)
    taulist = np.zeros(7)
    # print("Mlist",np.array(self._Ms).shape)
    # print("Slist", np.array(self._Ses).shape)
    # print("Glist", np.array(self.Glist).shape)
    for i in range(7):
        Mi = np.dot(Mi, Mlist[i])
        Ai[:, i] = np.dot(mr.Adjoint(mr.TransInv(Mi)), Slist[:, i])
        # qs 是 thetalist
        AdTi[i] = mr.Adjoint(np.dot(mr.MatrixExp6(mr.VecTose3(Ai[:, i] * -qs[i])), mr.TransInv(Mlist[i])))
        # dqs 是 dthetalist
        Vi[:, i + 1] = np.dot(AdTi[i], Vi[:, i]) + Ai[:, i] * dqs[i]
        Vri[:, i + 1] = np.dot(AdTi[i], Vri[:, i]) + Ai[:, i] * dqrs[i]
        # ddqrs 是ddthetalist
        Vdri[:, i + 1] = np.dot(AdTi[i], Vdri[:, i]) + Ai[:, i] * ddqrs[i] + np.dot(mr.ad(Vi[:, i + 1]), Ai[:, i]) * \
                         dqrs[i]

    for i in range(7 - 1, -1, -1):
        Fi = np.array(AdTi[i + 1]).T @ Fi + Glist[i] @ Vdri[:, i + 1] + 0.5 * (
                -mr.ad(Vi[:, i + 1]).T @ (Glist[i] @ Vri[:, i + 1])
                - mr.ad(Vri[:, i + 1]).T @ (Glist[i] @ Vi[:, i + 1])
                + Glist[i] @ (mr.ad(Vri[:, i + 1]) @ Vi[:, i + 1])
        )

        taulist[i] = np.dot(np.array(Fi).T, Ai[:, i])

    return taulist
q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
q_dot = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
qr_dot = np.array([0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72])
qr_ddot = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
print(inv_dynamics_adaptive(q, q_dot, qr_dot, qr_ddot))
Y = get_adaptive_identification_matrix(q, q_dot, qr_dot, qr_ddot)
print(Y@inertial_parameters)