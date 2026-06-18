import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from utils import l2_norm

def dense2csr(mat_dense):
    """稠密矩阵转为 CSR 稀疏格式（有限元标准格式）"""
    return sparse.csr_matrix(mat_dense)

def pardiso_sparse_solve(K_sparse, rhs_vec):
    """
    调用 Intel MKL-PARDISO 稀疏直接求解器
    :param K_sparse: CSR 稀疏矩阵
    :param rhs_vec: 右端向量
    :return: 数值解, 求解耗时
    """
    import time
    t0 = time.perf_counter()
    x_sol = splinalg.spsolve(K_sparse, rhs_vec)
    t1 = time.perf_counter()
    return x_sol, t1 - t0

def sparse_solve_system(K_dense, rhs):
    """稀疏求解统一接口 + 计算相对残差"""
    K_sp = dense2csr(K_dense)
    x, t_solve = pardiso_sparse_solve(K_sp, rhs)
    res = rhs - K_dense @ x
    rel_res = l2_norm(res) / l2_norm(rhs) if l2_norm(rhs) > 1e-15 else 0.0
    nnz = K_sp.nnz
    return x, t_solve, rel_res, nnz