import numpy as np

# ===================== 一维杆单元 =====================
def bar_elem_stiffness(E, A, length):
    """一维杆单元刚度矩阵 (2×2)"""
    k = E * A / length
    ke = np.array([
        [k, -k],
        [-k, k]
    ], dtype=np.float64)
    return ke

def assemble_global_K(n_node, elem_nodes, LM_matrix, elem_ke_list):
    """
    组装总体刚度矩阵
    :param n_node: 总节点数
    :param elem_nodes: 单元节点编号列表
    :param LM_matrix: 对号矩阵 LM
    :param elem_ke_list: 每个单元的刚度矩阵
    :return: 总体刚度矩阵 K
    """
    total_dof = n_node
    K_global = np.zeros((total_dof, total_dof), dtype=np.float64)
    for lm, ke in zip(LM_matrix, elem_ke_list):
        for i in range(2):
            for j in range(2):
                K_global[lm[i], lm[j]] += ke[i, j]
    return K_global

def reduce_system(K, F, known_dof, known_disp):
    """
    边界条件处理：自由度缩减
    输入: 总K、总载荷F、已知自由度、已知位移
    输出: K_FF, rhs, free_dof, fixed_dof
    rhs = f_F - K_EF^T * d_E
    """
    n = K.shape[0]
    free_dof = [idx for idx in range(n) if idx not in known_dof]
    fixed_dof = known_dof

    K_FF = K[np.ix_(free_dof, free_dof)]
    K_EF = K[np.ix_(fixed_dof, free_dof)]
    f_F = F[free_dof]
    d_E = np.array(known_disp, dtype=np.float64)

    rhs = f_F - K_EF.T @ d_E
    return K_FF, rhs, free_dof, fixed_dof

def recover_full_disp(free_dof, fixed_dof, d_F, known_disp, total_dof):
    """由未知位移重构全场位移"""
    d_full = np.zeros(total_dof, dtype=np.float64)
    d_full[free_dof] = d_F
    d_full[fixed_dof] = known_disp
    return d_full

def calc_reaction_force(K, d_full, F_total, fixed_dof):
    """计算约束反力"""
    reaction = K @ d_full - F_total
    return reaction[fixed_dof]

def calc_bar_axial_force(elem_ke, elem_disp):
    """由单元两端位移计算杆单元轴力"""
    force = elem_ke @ elem_disp
    return force[0]

# ===================== 作业算例0-1：一维两单元杆结构 =====================
def truss_case_1d_two_bar():
    """
    一维两杆桁架标准算例
    K = [[100, -100, 0],
         [-100, 300, -200],
         [0, -200, 200]]
    约束: 节点0位移=0；载荷: 节点2 外力=10
    理论解: d1=0.1, d2=0.15
    """
    K = np.array([
        [100.0, -100.0, 0.0],
        [-100.0, 300.0, -200.0],
        [0.0, -200.0, 200.0]
    ], dtype=np.float64)
    F_total = np.array([0.0, 0.0, 10.0], dtype=np.float64)
    known_dof = [0]
    known_disp = [0.0]
    n_node = 3

    # 单元信息（两杆单元）
    elem_list = [[0, 1], [1, 2]]
    LM_list = [[0, 1], [1, 2]]
    ke1 = bar_elem_stiffness(E=100, A=1, length=1)
    ke2 = bar_elem_stiffness(E=200, A=1, length=1)
    ke_list = [ke1, ke2]

    return K, F_total, known_dof, known_disp, n_node, elem_list, LM_list, ke_list