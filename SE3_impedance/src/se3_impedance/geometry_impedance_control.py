import numpy as np

class geometry_impedance_control:
    def __init__(self):
        pass
    
    def skew(self, w):
        """生成三维向量 w 的反对称矩阵"""
        return np.array([[0, -w[2], w[1]],
                         [w[2], 0, -w[0]],
                         [-w[1], w[0], 0]])
    
    def _bracket(self, a, b):

        skew_a = self.skew(a)
        skew_b = self.skew(b)
        return skew_a @ skew_b + skew_b @ skew_a
    
    def exponential_map_so3(self, xi):

        I_3 = np.eye(3)
        rotation_angle = np.linalg.norm(xi)

        # 3. 增加奇异点保护：如果角度极其接近0，直接返回单位阵
        if rotation_angle < 1e-7:
            return np.eye(3)
        
        # 你的数学计算逻辑是完全正确的
        s = 2 * np.sin(0.5 * rotation_angle) / rotation_angle
        c = np.cos(0.5 * rotation_angle)
        alpha = s * c
        beta = s * s
        skew_xi = self.skew(xi)

        # 2. 修正矩阵乘法，将 * 替换为 @ (矩阵乘法运算符)
        R = I_3 + alpha * skew_xi + 0.5 * beta * (skew_xi @ skew_xi)
        return R

    def dexp_xi(self, xi):

        
        norm_xi = np.linalg.norm(xi)
        I_3 = np.eye(3)
        
        # 奇异点保护：当旋转向量极小时，dexp 退化为单位矩阵
        if norm_xi < 1e-7:
            return I_3
            
        # 复用你之前的 s 和 c 的计算逻辑
        s = 2 * np.sin(0.5 * norm_xi) / norm_xi
        c = np.cos(0.5 * norm_xi)
        
        # 根据公式 (9a) 和 (9b) 计算 alpha 和 beta
        alpha = s * c
        beta = s * s
        
        skew_xi = self.skew(xi)
        skew_xi_sq = skew_xi @ skew_xi
        
        # 公式 (26)
        dexp_matrix = I_3 + 0.5 * beta * skew_xi + ((1.0 - alpha) / (norm_xi ** 2)) * skew_xi_sq
        
        return dexp_matrix
        
    def dexp_xi_inv(self, xi):

        theta = np.linalg.norm(xi)  
        I_3 = np.eye(3)
        
        # 奇异点保护：当旋转极小时，退化为单位矩阵
        if self.NearZero(theta):
            return I_3
        
        skew_xi = self.skew(xi)
        skew_xi_sq = skew_xi @ skew_xi
        
        # 根据文章公式 (80)，gamma = (theta / 2) * cot(theta / 2)
        # 我们直接将 (1 - gamma) / theta^2 展开化简，避免数值计算问题：
        factor = (1.0 / theta - 0.5 / np.tan(theta / 2.0)) / theta
        
        # 对应公式 (27)
        dexp_inv_matrix = I_3 - 0.5 * skew_xi + factor * skew_xi_sq
        
        return dexp_inv_matrix
    
    def exponential_map_se3(self, lamda):

        eta = lamda[0:3]
        xi = lamda[3:6]
        rotation_part = self.exponential_map_so3(xi)
        # 1. 使用 26式的公式，仅通过 xi 算出 3x3 的矩阵
        dexp_matrix = self.dexp_xi(xi) 

        # 2. 对应 10式的右上角，将算出的矩阵与 eta 相乘
        translation_part = dexp_matrix @ eta
        T = np.eye(4)
        T[:3,:3] = rotation_part
        T[:3,3] = translation_part

        return T

    def NearZero(self, z):

        return abs(z) < 1e-6
        
    def log_map_so3(self, R):

        acosinput = (np.trace(R) - 1) / 2.0
                
        # 情况1：无旋转 (theta = 0)
        if acosinput >= 1:
            return np.zeros(3)
            
        # 情况2：旋转 180 度 (theta = pi) 奇异点处理
        elif acosinput <= -1:
            # 使用 NearZero 判断对角线元素，找出不接近 0 的分量来求旋转轴
            if not self.NearZero(1 + R[2, 2]):
                omg = (1.0 / np.sqrt(2 * (1 + R[2, 2]))) * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
            elif not self.NearZero(1 + R[1, 1]):
                omg = (1.0 / np.sqrt(2 * (1 + R[1, 1]))) * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
            else:
                omg = (1.0 / np.sqrt(2 * (1 + R[0, 0]))) * np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
            
            return np.pi * omg
            
        # 情况3：常规情况 (0 < theta < pi)
        else:
            theta = np.arccos(acosinput)
            skew_matrix = (theta / (2.0 * np.sin(theta))) * (R - R.T)
            # 从反对称矩阵中提取出 3D 向量 xi
            xi = np.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])
            return xi

    def log_map_se3(self, T):

        # 提取旋转矩阵 R 和平移向量 r
        R = T[:3, :3]
        r = T[:3, 3]
        
        # 1. 提取旋转向量 xi (直接复用前面的对数映射)
        xi = self.log_map_so3(R)
        theta = np.linalg.norm(xi)
        
        # 2. 奇异点保护：无旋转时，平移分量直接就是位置偏差 r
        if self.NearZero(theta):
            eta = r
        else:
            # 3. 计算 dexp_{xi}^{-1}，对应文章公式 (27)
            dexp_inv = self.dexp_xi_inv(xi)
            
            # 4. 计算线性的指数坐标 eta
            eta = dexp_inv @ r
            
        # 5. 拼接成 6 维向量 lamda = [eta^T, xi^T]^T
        lamda = np.concatenate((eta, xi))
        
        return lamda
    
    def ceiling_form(self, vec):


        vec = np.asarray(vec).flatten()
        
        if len(vec) == 3:
            # 对应公式 (1)：3D 向量转 3x3 反对称矩阵 [cite: 124]
            return self.skew(vec)
            
        elif len(vec) == 6:
            # 对应公式 (2)：6D 向量转 4x4 矩阵 [cite: 124]
            # 根据论文约定 V = (v^T, ω^T)^T 或 λ = (η^T, ξ^T)^T [cite: 125, 134]
            # 前三维是线速度(平移)，后三维是角速度(旋转)
            v = vec[0:3]
            omega = vec[3:6]
            
            mat = np.zeros((4, 4))
            mat[:3, :3] = self.skew(omega)
            mat[:3, 3] = v
            # 第四行保持为 [0, 0, 0, 0]
            
            return mat
            
        else:
            raise ValueError("输入向量的维度必须是 3 或 6！")

    def Ad_T(self, T):

        R = T[:3, :3]
        r = T[:3, 3]
        
        skew_r = self.skew(r)
        zeros_3 = np.zeros((3, 3))
        
        # 对应公式 (5)
        # 注意：因为论文中定义 V = [v^T, ω^T]^T (平移在前，旋转在后)
        # 所以 Ad_T 的分块矩阵形式为：
        # [ R    [r]R ]
        # [ 0_3   R   ]
        AdT = np.block([
            [R,       skew_r @ R],
            [zeros_3, R         ]
        ])
        
        return AdT

    def ad_V(self, V):

        
        V = np.asarray(V).flatten()
        v = V[0:3]
        omega = V[3:6]
        
        skew_v = self.skew(v)
        skew_omega = self.skew(omega)
        zeros_3 = np.zeros((3, 3))
        
        # 对应公式 (3)
        # [ [ω]   [v] ]
        # [ 0_3   [ω] ]
        adV = np.block([
            [skew_omega, skew_v    ],
            [zeros_3,    skew_omega]
        ])
        
        return adV

    def _C_xi(self, xi, eta):

        
        norm_xi = np.linalg.norm(xi)
        skew_eta = self.skew(eta)
        
        # 奇异点保护：当 xi -> 0 时，C_0(eta) = 1/2 * [eta] (见文章 31 式下方说明)
        if self.NearZero(norm_xi):
            return 0.5 * skew_eta
            
        skew_xi = self.skew(xi)
        skew_xi_sq = skew_xi @ skew_xi
        
        # 定义文章中的交换子 [eta, xi] = [eta][xi] + [xi][eta]
        skew_eta_xi = self._bracket(eta, xi)
        
        # 计算 alpha, beta
        s = 2.0 * np.sin(0.5 * norm_xi) / norm_xi
        c = np.cos(0.5 * norm_xi)
        alpha = s * c
        beta = s * s
        
        xi_dot_eta = np.dot(xi, eta)
        
        # 对应公式 (30) 的四项
        term1 = 0.5 * beta * skew_eta
        term2 = ((1.0 - alpha) / (norm_xi ** 2)) * skew_eta_xi
        term3 = ((alpha - beta) / (norm_xi ** 2)) * xi_dot_eta * skew_xi
        term4 = -(1.0 / (norm_xi ** 2)) * (3.0 * (1.0 - alpha) / (norm_xi ** 2) - 0.5 * beta) * xi_dot_eta * skew_xi_sq
        
        return term1 + term2 + term3 + term4

    def _D_xi(self, xi, eta):

        norm_xi = np.linalg.norm(xi)
        skew_eta = self.skew(eta)
        
        # 奇异点保护：当 xi -> 0 时，D_0(eta) = -1/2 * [eta] (见文章 31 式下方说明)
        if self.NearZero(norm_xi):
            return -0.5 * skew_eta
            
        skew_xi = self.skew(xi)
        skew_xi_sq = skew_xi @ skew_xi
        # 直接调用封装好的 _bracket 函数
        skew_eta_xi = self._bracket(eta, xi)
        
        # 计算 gamma 和 beta
        beta = 2.0 * (1.0 - np.cos(norm_xi)) / (norm_xi ** 2)
        gamma = (norm_xi / 2.0) / np.tan(norm_xi / 2.0)
        
        # 我们之前推导过的 (1 - gamma) / ||xi||^2
        factor_gamma = (1.0 / norm_xi - 0.5 / np.tan(norm_xi / 2.0)) / norm_xi
        
        xi_dot_eta = np.dot(xi, eta)
        
        # 对应公式 (31) 的三项
        term1 = -0.5 * skew_eta
        term2 = factor_gamma * skew_eta_xi
        term3 = (1.0 / (norm_xi ** 2)) * ((1.0 / beta + gamma - 2.0) / (norm_xi ** 2)) * xi_dot_eta * skew_xi_sq
        
        return term1 + term2 + term3

    def dexp_se3(self, lamda):

        
        eta = lamda[0:3]
        xi = lamda[3:6]
        
        # 复用之前写好的 SO(3) 上的 dexp_xi
        dexp_xi_mat = self.dexp_xi(xi) 
        
        C_mat = self._C_xi(xi, eta)

        zeros_3 = np.zeros((3, 3))
        
        # 公式 (28) 的分块矩阵组装
        dexp_lam = np.block([
            [dexp_xi_mat, C_mat      ],
            [zeros_3,     dexp_xi_mat]
        ])
        
        return dexp_lam

    def dexp_inv_se3(self, lamda):

        
        eta = lamda[0:3]
        xi = lamda[3:6]
        
        # 复用之前写好的 SO(3) 上的逆微分 dexp_xi_inv
        dexp_inv_xi_mat = self.dexp_xi_inv(xi)
        D_mat = self._D_xi(xi, eta)
        zeros_3 = np.zeros((3, 3))
        
        # 公式 (29) 的分块矩阵组装
        dexp_inv_lam = np.block([
            [dexp_inv_xi_mat, D_mat          ],
            [zeros_3,         dexp_inv_xi_mat]
        ])
        
        return dexp_inv_lam

    def dexp_dot_so3(self, xi, xi_dot):

        return self._C_xi(xi, xi_dot)

    def dexp_inv_dot_so3(self, xi, xi_dot):

        return self._D_xi(xi, xi_dot)

    def calc_body_angular_acceleration(self, xi, xi_dot, xi_ddot):

        minus_xi = -xi
        minus_xi_dot = -xi_dot
        
        term1 = self.dexp_xi(minus_xi) @ xi_ddot
        term2 = self.dexp_dot_so3(minus_xi, minus_xi_dot) @ xi_dot
        
        return term1 + term2


    def _C_dot_xi(self, xi, xi_dot, eta, eta_dot):

        norm_xi = np.linalg.norm(xi)
        if self.NearZero(norm_xi):
            # 公式 (42)
            return 0.5 * self.skew(eta_dot) + (1.0 / 6.0) * self._bracket(xi_dot, eta)
            
        skew_xi = self.skew(xi)
        skew_xi_sq = skew_xi @ skew_xi
        xi_dot_xi = np.dot(xi, xi_dot)
        
        s = 2.0 * np.sin(0.5 * norm_xi) / norm_xi
        c = np.cos(0.5 * norm_xi)
        alpha, beta = s * c, s * s
        
        # 公式 (40a, 40b, 40c)
        Gamma_1 = (1.0 - alpha) / (norm_xi ** 2)
        Gamma_2 = (alpha - beta) / (norm_xi ** 2)
        Gamma_3 = (0.5 * beta - 3.0 * Gamma_1) / (norm_xi ** 2)
        
        zeta_eta = np.dot(xi, eta) * xi_dot_xi / (norm_xi ** 2)
        
        # 公式 (41a, 41b)
        delta_0 = np.dot(xi_dot, eta) + np.dot(xi, eta_dot)
        delta_1 = np.dot(xi, eta) * self.skew(xi_dot) + np.dot(xi, xi_dot) * self.skew(eta) + \
                  (delta_0 - 4.0 * zeta_eta) * skew_xi
        delta_2 = np.dot(xi, eta) * self._bracket(xi, xi_dot) + np.dot(xi, xi_dot) * self._bracket(xi, eta) + \
                  (delta_0 - 5.0 * zeta_eta) * skew_xi_sq
                  
        # 公式 (38)
        term1 = 0.5 * beta * (self.skew(eta_dot) - zeta_eta * skew_xi)
        term2 = Gamma_1 * (self._bracket(eta_dot, xi) + self._bracket(eta, xi_dot) + zeta_eta * skew_xi)
        term3 = Gamma_2 * (delta_1 + zeta_eta * skew_xi_sq)
        term4 = Gamma_3 * delta_2
        
        return term1 + term2 + term3 + term4

    def _D_dot_xi(self, xi, xi_dot, eta, eta_dot):

        norm_xi = np.linalg.norm(xi)
        if self.NearZero(norm_xi):
            # 公式 (43)
            return -0.5 * self.skew(eta_dot) + (1.0 / 12.0) * self._bracket(xi_dot, eta)
            
        skew_xi = self.skew(xi)
        skew_xi_sq = skew_xi @ skew_xi
        xi_dot_xi = np.dot(xi, xi_dot)
        
        beta = 2.0 * (1.0 - np.cos(norm_xi)) / (norm_xi ** 2)
        gamma = (norm_xi / 2.0) / np.tan(norm_xi / 2.0)
        
        # 公式 (40d, 40e)
        Gamma_4 = (1.0 - gamma) / (norm_xi ** 2)
        Gamma_5 = ((1.0 / beta) + gamma - 2.0) / (norm_xi ** 4)
        
        zeta_eta = np.dot(xi, eta) * xi_dot_xi / (norm_xi ** 2)
        
        # 公式 (41c)
        delta_0 = np.dot(xi_dot, eta) + np.dot(xi, eta_dot)
        delta_3 = np.dot(xi, eta) * self._bracket(xi, xi_dot) + np.dot(xi, xi_dot) * self._bracket(xi, eta) + \
                  (delta_0 - 3.0 * zeta_eta) * skew_xi_sq
                  
        # 公式 (39)
        term1 = -0.5 * self.skew(eta_dot)
        term2 = (2.0 / (norm_xi ** 2)) * ((1.0 - gamma / beta) / (norm_xi ** 2)) * zeta_eta * skew_xi_sq
        term3 = Gamma_4 * (self._bracket(eta_dot, xi) + self._bracket(eta, xi_dot))
        term4 = Gamma_5 * delta_3
        
        return term1 + term2 + term3 + term4

    def dexp_dot_se3(self, lamda, lamda_dot):

        eta, xi = lamda[0:3], lamda[3:6]
        eta_dot, xi_dot = lamda_dot[0:3], lamda_dot[3:6]
        
        C_xi_dot_mat = self._C_xi(xi, xi_dot)
        C_dot_mat = self._C_dot_xi(xi, xi_dot, eta, eta_dot)
        zeros_3 = np.zeros((3, 3))
        
        return np.block([
            [C_xi_dot_mat, C_dot_mat   ],
            [zeros_3,      C_xi_dot_mat]
        ])

    def dexp_inv_dot_se3(self, lamda, lamda_dot):

        eta, xi = lamda[0:3], lamda[3:6]
        eta_dot, xi_dot = lamda_dot[0:3], lamda_dot[3:6]
        
        D_xi_dot_mat = self._D_xi(xi, xi_dot)
        D_dot_mat = self._D_dot_xi(xi, xi_dot, eta, eta_dot)
        zeros_3 = np.zeros((3, 3))
        
        return np.block([
            [D_xi_dot_mat, D_dot_mat   ],
            [zeros_3,      D_xi_dot_mat]
        ])

    def calc_body_twist_acceleration(self, lamda, lamda_dot, lamda_ddot):

        minus_lam = -lamda
        minus_lam_dot = -lamda_dot
        
        term1 = self.dexp_se3(minus_lam) @ lamda_ddot
        term2 = self.dexp_dot_se3(minus_lam, minus_lam_dot) @ lamda_dot
        
        return term1 + term2

if __name__ == "__main__":
    gic = geometry_impedance_control()
    print(gic.exponential_map_so3(np.array([1, 2, 3])))