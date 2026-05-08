import numpy as np

class SE3ImpedanceController:
    def __init__(self, gic_instance):
        """
        初始化控制器
        :param gic_instance: 传入已经实例化的 geometry_impedance_control 对象
        """
        self.gic = gic_instance
        
        # =========================================================
        # 【物理优先的阻抗参数配置】
        # 物理空间的惯量 A 和阻尼 D，以及数学空间的刚度 K_lam
        # =========================================================
        
        # 1. 期望的物理表观惯量矩阵 A (常数，对角阵)
        # 决定机器人在真实世界受力时的“质量手感”
        self.A = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]) 
        
        # 2. 期望的李代数刚度矩阵 K_lambda (常数，对角阵)
        # 必须定在平直空间，保证完美的线性抛物线势能，消灭 180 度死锁
        self.K_lam = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])* 0.05
        
        # 3. 期望的物理阻尼矩阵 D (常数，对角阵)
        # 保证物理空间交互能量的耗散，这里设为物理空间的临界阻尼
        self.D = 2.0 * np.sqrt(self.A @ self.K_lam) 
        
        # =========================================================
        # 零空间控制参数 (防止冗余自由度乱晃)
        # =========================================================
        self.q_nominal = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        self.k_null = 10.0   
        self.d_null = 2.0    

    def compute_torque(self, 
                        q, q_dot,                   
                        T, V,                       
                        J, M, qfrc_bias, J_dot_q_dot, 
                        T_d, V_d, V_d_dot,          
                        F_ext):                     
            """
            计算阻抗控制下发给关节的电机力矩 (严格论文理论修正版)
            """
            

            T_tilde = np.linalg.inv(T) @ T_d
            lamda = self.gic.log_map_se3(T_tilde)
            
            Ad_T_tilde = self.gic.Ad_T(T_tilde)
            V_tilde = Ad_T_tilde @ V_d - V
            
            dexp_inv_lam = self.gic.dexp_inv_se3(lamda)
            lamda_dot = dexp_inv_lam @ V_tilde
            

            dexp_lam = self.gic.dexp_se3(lamda)
            

            d_dexp_lam = self.gic.dexp_dot_se3(lamda, lamda_dot) 
            

            A_lam_dynamic = dexp_lam.T @ self.A @ dexp_lam
            

            D_lam_dynamic = (dexp_lam.T @ self.D @ dexp_lam) + \
                            (dexp_lam.T @ self.A @ d_dexp_lam)
            

            gamma = dexp_lam.T @ (-F_ext) 
            

            lamda_ddot_ref = np.linalg.inv(A_lam_dynamic) @ (gamma - D_lam_dynamic @ lamda_dot - self.K_lam @ lamda)
            

            V_tilde_dot = dexp_lam @ lamda_ddot_ref + d_dexp_lam @ lamda_dot
            
            ad_V_tilde = self.gic.ad_V(V_tilde)

            V_ref_dot = Ad_T_tilde @ V_d_dot - V_tilde_dot + ad_V_tilde @ V
            
            # ---------------------------------------------------------
            # 第六步：逆动力学求解与零空间投影 (你的实现非常完美，无需修改)
            # ---------------------------------------------------------
            M_inv = np.linalg.inv(M)
            Lambda_inv = J @ M_inv @ J.T   
            
            # 奇异点保护
            Lambda_matrix = np.linalg.inv(Lambda_inv + 1e-5 * np.eye(6))
            
            # 动态一致性伪逆
            J_Mpinv = M_inv @ J.T @ Lambda_matrix 
            
            # 计算任务力矩 Eq 65 & 66
            tau_task = J.T @ Lambda_matrix @ (V_ref_dot - J_dot_q_dot) + qfrc_bias - J.T @ F_ext
            
            # 计算零空间力矩
            N_tau = np.eye(7) - J.T @ J_Mpinv.T 
            tau_0 = -self.k_null * (q - self.q_nominal) - self.d_null * q_dot
            tau_null = N_tau @ tau_0
            
            # 终极下发力矩
            tau = tau_task + tau_null
            
            return tau