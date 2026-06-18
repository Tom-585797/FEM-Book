import numpy as np
from utils import calc_residual

def ldlt_factor(mat):
    """
    对称方阵 LDL^T 分解: mat = L @ D @ L.T
    :param mat: 输入对称矩阵 (n,n)，浮点型
    :return: L(单位下三角), D(对角向量), success(分解是否成功)
    下标: 0起始；检测非正主元并报错
    """
    n = mat.shape[0]
    A = np.array(mat, dtype=np.float64, copy=True)
    L = np.eye(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)
    eps = 1e-12

    for j in range(n):
        # 计算对角元 D[j]
        sum_d = 0.0
        for k in range(j):
            sum_d += L[j, k] * L[j, k] * D[k]
        D[j] = A[j, j] - sum_d

        # 检测零主元 / 负主元
        if D[j] <= eps:
            print(f"[分解失败] 第 {j} 个主元 D[{j}] = {D[j]:.4e}，矩阵非正定/奇异！")
            return None, None, False

        # 计算 L 第j列下方元素
        for i in range(j + 1, n):
            sum_l = 0.0
            for k in range(j):
                sum_l += L[i, k] * L[j, k] * D[k]
            L[i, j] = (A[i, j] - sum_l) / D[j]

    return L, D, True

def ldlt_solve(L, D, rhs):
    """
    求解方程组 L D L^T x = rhs
    三步: 前代 -> 对角求解 -> 回代
    :param L: 单位下三角阵
    :param D: 对角向量
    :param rhs: 右端向量
    :return: 解向量 x
    """
    n = len(rhs)
    b = np.array(rhs, dtype=np.float64).reshape(-1, 1)

    # 1. 前代: L * y = b
    y = np.zeros_like(b)
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i, k] * y[k, 0]
        y[i, 0] = b[i, 0] - s

    # 2. 对角求解: D * z = y
    z = np.zeros_like(y)
    for i in range(n):
        z[i, 0] = y[i, 0] / D[i]

    # 3. 回代: L^T * x = z
    x = np.zeros_like(z)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for k in range(i + 1, n):
            s += L[k, i] * x[k, 0]
        x[i, 0] = z[i, 0] - s

    return x.ravel()

def solve_equilibrium(K_FF, rhs, method="ldlt", **options):
    """
    作业标准统一求解接口
    求解缩减平衡方程: K_FF * d_F = rhs
    :param K_FF: 缩减刚度矩阵
    :param rhs: 右端项 f_F - K_EF^T * d_E
    :param method: 求解方法，仅实现 ldlt
    :return: d_F(位移解), res_norm(残差范数), success(状态)
    """
    if method.lower() == "ldlt":
        L, D, flag = ldlt_factor(K_FF)
        if not flag:
            return None, -1.0, False
        x = ldlt_solve(L, D, rhs)
        _, res_norm = calc_residual(K_FF, x, rhs)
        return x, res_norm, True
    else:
        raise NotImplementedError("当前仅实现稠密 LDL^T 求解器")