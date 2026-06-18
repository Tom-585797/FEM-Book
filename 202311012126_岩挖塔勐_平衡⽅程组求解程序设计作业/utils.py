import numpy as np
import time

def l2_norm(vec):
    """计算向量2-范数"""
    return np.linalg.norm(vec, ord=2)

def calc_residual(K, x, rhs):
    """计算残差向量 & 残差2-范数"""
    res = rhs - K @ x
    res_norm = l2_norm(res)
    return res, res_norm

def relative_residual(K, x, rhs):
    """相对残差 ||r|| / ||R||"""
    _, r_norm = calc_residual(K, x, rhs)
    rhs_norm = l2_norm(rhs)
    if rhs_norm < 1e-15:
        return 0.0
    return r_norm / rhs_norm

def relative_error(x_num, x_exact):
    """解的相对误差 ||x_num - x_exact|| / ||x_exact||"""
    diff = x_num - x_exact
    diff_norm = l2_norm(diff)
    exact_norm = l2_norm(x_exact)
    if exact_norm < 1e-15:
        return 0.0
    return diff_norm / exact_norm

def matrix_condition_number(K):
    """计算矩阵2-条件数"""
    return np.linalg.cond(K, p=2)

def timeit_func(func, *args, repeat=3):
    """多次运行取平均耗时，消除偶然误差"""
    time_records = []
    result = None
    for _ in range(repeat):
        t_start = time.perf_counter()
        result = func(*args)
        t_end = time.perf_counter()
        time_records.append(t_end - t_start)
    avg_time = np.mean(time_records)
    return result, avg_time