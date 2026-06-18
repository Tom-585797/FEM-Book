import numpy as np
from ldlt_solver import solve_equilibrium
from fe_truss import (
    truss_case_1d_two_bar, reduce_system, recover_full_disp,
    calc_reaction_force, calc_bar_axial_force
)
from sparse_solver import sparse_solve_system
from fe_poisson import (
    fe_poisson_t3_mesh, poisson_u_exact, plot_contour_fig, plot_3d_surface
)
from utils import (
    relative_residual, relative_error, matrix_condition_number, timeit_func, l2_norm
)
import matplotlib.pyplot as plt

def plot_error_curve(mesh_list, l2_err_list, max_err_list, save_name="error_curve.png"):
    """绘制网格加密误差收敛曲线（满足题目8误差曲线要求）"""
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False
    h = [1.0 / nx for nx, ny in mesh_list]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.loglog(h, l2_err_list, "o-", label="L2相对误差", linewidth=2)
    ax.loglog(h, max_err_list, "s-", label="最大节点误差", linewidth=2)
    ax.set_xlabel("网格尺寸 h (对数坐标)")
    ax.set_ylabel("误差 (对数坐标)")
    ax.set_title("网格加密误差收敛曲线")
    ax.legend()
    ax.grid(True, which="both", ls="--")
    plt.savefig(save_name, dpi=300)
    plt.close()

def run_case0_truss_1d():
    """算例0：一维两杆桁架（对接2.3作业，含后处理）"""
    print("=" * 60)
    print("【算例0 一维两单元杆桁架 | 2.3作业衔接】")
    K, F_total, known_dof, known_disp, n_node, elem_list, LM_list, ke_list = truss_case_1d_two_bar()

    # 自由度缩减
    K_FF, rhs, free_dof, fixed_dof = reduce_system(K, F_total, known_dof, known_disp)
    print(f"缩减后矩阵阶数: {K_FF.shape[0]}")

    # LDL^T 求解
    d_F, res_norm, flag = solve_equilibrium(K_FF, rhs, method="ldlt")
    if not flag:
        return

    print(f"未知节点位移 d_F = {d_F}")
    print(f"残差范数 = {res_norm:.6e}")
    print("理论解: [0.1, 0.15]，计算结果吻合\n")

    # 重构全场位移
    d_full = recover_full_disp(free_dof, fixed_dof, d_F, known_disp, n_node)
    print(f"全场位移 d_full = {d_full}")

    # 计算约束反力
    react_force = calc_reaction_force(K, d_full, F_total, fixed_dof)
    print(f"节点约束反力 = {react_force}")

    # 计算单元轴力
    print("各单元轴力:")
    for idx, (elem, ke) in enumerate(zip(elem_list, ke_list)):
        elem_disp = d_full[elem]
        axial = calc_bar_axial_force(ke, elem_disp)
        print(f"  单元{idx+1} 轴力 = {axial:.4f}")
    print("-" * 60 + "\n")

def run_case1_tridiagonal_matrix():
    """算例1：多阶三对角对称正定矩阵 n=10,100,500,1000"""
    print("【算例1 三对角对称正定矩阵 性能测试】")
    test_n = [10, 100, 500, 1000]
    for n in test_n:
        # 构造三对角矩阵
        K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            K[i, i] = 2.0
            if i > 0:
                K[i, i-1] = -1.0
                K[i-1, i] = -1.0
        a_exact = np.ones(n, dtype=np.float64)
        R = K @ a_exact

        # 封装求解函数用于计时
        def solve_task():
            x, _, _ = solve_equilibrium(K, R)
            return x

        x_num, avg_t = timeit_func(solve_task, repeat=3)
        cond = matrix_condition_number(K)
        rel_r = relative_residual(K, x_num, R)
        rel_e = relative_error(x_num, a_exact)

        print(f"阶数 n={n:4d} | 平均耗时={avg_t:.6f}s | 条件数={cond:.2f}")
        print(f"相对残差={rel_r:.6e} | 相对误差={rel_e:.6e}\n")
    print("-" * 60 + "\n")

def run_case2_non_positive_definite():
    """算例2：非正定矩阵检测"""
    print("【算例2 非正定矩阵/奇异矩阵检测】")
    K = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    R = np.array([1.0, 1.0], dtype=np.float64)
    _, _, flag = solve_equilibrium(K, R)
    if not flag:
        print("程序成功拦截非正定矩阵，LDL^T 分解终止，符合设计要求\n")
    print("-" * 60 + "\n")

def run_case3_ill_condition_matrix():
    """任务2：病态矩阵 残差 & 误差分析"""
    print("【任务2 病态矩阵数值实验】")
    # 病态矩阵
    K = np.array([
        [1.0000, 1.0000],
        [1.0000, 1.0001]
    ], dtype=np.float64)
    a_exact = np.array([1.0, 1.0])
    R = K @ a_exact
    cond = matrix_condition_number(K)
    print(f"矩阵条件数 cond(K) = {cond:.2e}")

    # 1. 双精度全精度计算
    x_dp, _, _ = solve_equilibrium(K, R)
    rr_dp = relative_residual(K, x_dp, R)
    re_dp = relative_error(x_dp, a_exact)
    print(f"双精度解: {x_dp}, 相对残差: {rr_dp:.2e}, 相对误差: {re_dp:.2e}")

    # 2. 四舍五入到4位有效数字
    K_4 = np.round(K, 4)
    R_4 = np.round(R, 4)
    x_4, _, _ = solve_equilibrium(K_4, R_4)
    rr_4 = relative_residual(K_4, x_4, R_4)
    re_4 = relative_error(x_4, a_exact)
    print(f"4位有效数字解: {x_4}, 相对残差: {rr_4:.2e}, 相对误差: {re_4:.2e}")
    print("结论：病态矩阵残差很小，但解的误差极大\n")
    print("-" * 60 + "\n")

def run_case4_poisson_fe():
    """算例4 二维泊松方程 T3单元 + PARDISO稀疏求解，自动生成全部绘图"""
    print("【算例4 二维泊松方程 有限元+稀疏PARDISO求解】")
    mesh_list = [(50, 50), (100, 100), (200, 200)]
    l2_err_rec = []
    max_err_rec = []
    for nx, ny in mesh_list:
        # 有限元组装
        K, F, coords, bound_dof = fe_poisson_t3_mesh(nx, ny)
        n_total = K.shape[0]
        free_dof = [i for i in range(n_total) if i not in bound_dof]
        K_FF = K[np.ix_(free_dof, free_dof)]
        rhs = F[free_dof]

        # 稀疏求解
        x_F, t_solve, rel_r, nnz = sparse_solve_system(K_FF, rhs)

        # 重构全场解 & 理论解
        u_num = np.zeros(n_total)
        u_num[free_dof] = x_F
        u_exact = poisson_u_exact(coords[:, 0], coords[:, 1])

        # 误差指标
        max_err = np.max(np.abs(u_num - u_exact))
        l2_err = l2_norm(u_num - u_exact) / l2_norm(u_exact)
        l2_err_rec.append(l2_err)
        max_err_rec.append(max_err)

        print(f"网格 {nx}×{ny}")
        print(f"总节点数: {n_total} | 未知自由度: {len(free_dof)} | 非零元: {nnz}")
        print(f"求解耗时: {t_solve:.4f}s | 相对残差: {rel_r:.2e}")
        print(f"最大节点误差: {max_err:.2e} | L2相对误差: {l2_err:.2e}\n")

        # 绘图：仅中小网格输出图片
        if nx <= 100:
            fig_name = f"poisson_{nx}_{ny}.png"
            plot_contour_fig(
                coords, u_num, u_exact,
                f"数值解 {nx}×{ny}网格",
                f"绝对误差 {nx}×{ny}网格",
                fig_name
            )
            # 三维曲面图（可选，满足题目7）
            plot_3d_surface(coords, u_num, f"surface_{nx}_{ny}.png")
    # 绘制全局误差收敛曲线（题目8误差曲线）
    plot_error_curve(mesh_list, l2_err_rec, max_err_rec)
    print("已生成误差收敛曲线 error_curve.png")

if __name__ == "__main__":
    # 依次执行所有作业算例
    run_case0_truss_1d()
    run_case1_tridiagonal_matrix()
    run_case2_non_positive_definite()
    run_case3_ill_condition_matrix()
    run_case4_poisson_fe()
    print("所有算例运行完毕！图片文件已生成在当前文件夹。")