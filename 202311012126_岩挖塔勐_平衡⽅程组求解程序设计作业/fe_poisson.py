import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# 理论解与右端项
def poisson_u_exact(x, y):
    """精确解 u = sin(πx)sin(πy)"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def poisson_f_source(x, y):
    """泊松方程右端项 f = 2π² sin(πx)sin(πy)"""
    return 2 * (np.pi ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y)

def fe_poisson_t3_mesh(nx, ny):
    """
    单位正方形 [0,1]×[0,1]，线性三角形单元 T3
    :param nx: x方向单元数
    :param ny: y方向单元数
    :return: K(稠密), F, 节点坐标, 边界自由度
    """
    # 生成网格节点
    x_line = np.linspace(0.0, 1.0, nx + 1)
    y_line = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(x_line, y_line)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    n_node = coords.shape[0]

    # 识别边界自由度（全域Dirichlet u=0）
    bound_dof = []
    tol = 1e-8
    for idx in range(n_node):
        xi, yi = coords[idx]
        if xi < tol or xi > 1-tol or yi < tol or yi > 1-tol:
            bound_dof.append(idx)

    # 单元划分：每个矩形分为2个三角单元
    elem_list = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i
            n3 = n2 + 1
            elem_list.append([n0, n1, n2])
            elem_list.append([n1, n3, n2])

    # 初始化总体矩阵与载荷向量
    K = np.zeros((n_node, n_node), dtype=np.float64)
    F = np.zeros(n_node, dtype=np.float64)

    # 单元循环 + 积分 + 组装
    for elem in elem_list:
        idx0, idx1, idx2 = elem
        x0, y0 = coords[idx0]
        x1, y1 = coords[idx1]
        x2, y2 = coords[idx2]

        # 三角形面积
        area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
        if area < 1e-12:
            continue

        # T3 单元刚度矩阵
        b1 = y1 - y2
        b2 = y2 - y0
        b3 = y0 - y1
        c1 = x2 - x1
        c2 = x0 - x2
        c3 = x1 - x0

        ke = (1.0 / (4.0 * area)) * np.array([
            [b1*b1 + c1*c1, b1*b2 + c1*c2, b1*b3 + c1*c3],
            [b2*b1 + c2*c1, b2*b2 + c2*c2, b2*b3 + c2*c3],
            [b3*b1 + c3*c1, b3*b2 + c3*c2, b3*b3 + c3*c3]
        ])

        # 单元载荷（重心单点积分）
        xg = (x0 + x1 + x2) / 3.0
        yg = (y0 + y1 + y2) / 3.0
        fg = poisson_f_source(xg, yg)
        fe = np.ones(3) * fg * area / 3.0

        # 整体组装
        for i in range(3):
            F[elem[i]] += fe[i]
            for j in range(3):
                K[elem[i], elem[j]] += ke[i, j]

    return K, F, coords, bound_dof

def plot_contour_fig(coords, u_num, u_exact, title1, title2, save_name):
    """绘制数值解云图 + 误差云图（同时满足7、8要求）"""
    x = coords[:, 0]
    y = coords[:, 1]
    err = np.abs(u_num - u_exact)

    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # 数值解
    cf1 = ax1.tricontourf(x, y, u_num, levels=60, cmap="jet")
    ax1.set_title(title1, fontsize=12)
    plt.colorbar(cf1, ax=ax1)
    # 误差
    cf2 = ax2.tricontourf(x, y, err, levels=60, cmap="jet")
    ax2.set_title(title2, fontsize=12)
    plt.colorbar(cf2, ax=ax2)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()

def plot_3d_surface(coords, u_num, save_name):
    """绘制三维曲面图（备选满足题目7）"""
    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False

    nx = int(np.sqrt(len(coords)) - 1)
    x = coords[:,0].reshape(nx+1, nx+1)
    y = coords[:,1].reshape(nx+1, nx+1)
    u = u_num.reshape(nx+1, nx+1)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, u, cmap="jet", linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    ax.set_title("泊松方程数值解三维曲面")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()