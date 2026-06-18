import numpy as np
import matplotlib.pyplot as plt

# 全局中文字体配置，彻底消除方框乱码
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

def alpha_supg(Pe):
    """SUPG自适应最优稳定参数"""
    if abs(Pe) < 1e-12:
        return 0.0
    coth = 1 / np.tanh(Pe)
    return coth - 1.0 / Pe

def element_matrix(kappa, v, le, alpha):
    """带人工稳定扩散的2节点线性单元刚度矩阵"""
    k_eff = kappa + alpha * v * le / 2.0
    # 扩散项矩阵
    K_diff = (k_eff / le) * np.array([[1, -1],
                                      [-1, 1]])
    # 对流项矩阵（非对称）
    K_adv = (v / 2.0) * np.array([[-1, 1],
                                  [-1, 1]])
    Ke = K_diff + K_adv
    return Ke

def solve_advection(nel, L, v, kappa, alpha):
    """求解对流扩散方程，返回节点坐标、数值解、解析解、总刚矩阵"""
    le = L / nel
    nnodes = nel + 1
    x = np.linspace(0, L, nnodes)
    K = np.zeros((nnodes, nnodes))
    F = np.zeros(nnodes)
    # 单元循环组装整体刚度矩阵
    for e_idx in range(nel):
        Ke = element_matrix(kappa, v, le, alpha)
        K[e_idx:e_idx+2, e_idx:e_idx+2] += Ke
    # 强制Dirichlet边界条件 θ(0)=0，θ(1)=1
    K[0, :] = 0.0
    K[:, 0] = 0.0
    K[0, 0] = 1.0
    F[0] = 0.0

    K[-1, :] = 0.0
    K[:, -1] = 0.0
    K[-1, -1] = 1.0
    F[-1] = 1.0
    # 线性方程组求解
    theta_num = np.linalg.solve(K, F)
    # 稳定计算解析精确解
    Pe_global = v * L / kappa
    theta_ex = np.expm1(v * x / kappa) / np.expm1(Pe_global)
    return x, theta_num, theta_ex, K

def max_abs_err(num_sol, ex_sol):
    """计算节点最大绝对误差"""
    return np.max(np.abs(num_sol - ex_sol))

def plot_solution_curve(Pe_target, nel=20, L=1.0, v=1.0):
    """绘制单Pe下三种格式解对比曲线"""
    le = L / nel
    kappa = v * le / (2 * Pe_target)
    alpha_gal = 0.0
    alpha_upw = 1.0
    alpha_spg = alpha_supg(Pe_target)
    # 分别求解三种离散格式
    x_gal, t_gal, tex, K_gal = solve_advection(nel, L, v, kappa, alpha_gal)
    x_upw, t_upw, _, _ = solve_advection(nel, L, v, kappa, alpha_upw)
    x_spg, t_spg, _, _ = solve_advection(nel, L, v, kappa, alpha_spg)
    # 计算各自最大误差
    err_gal = max_abs_err(t_gal, tex)
    err_upw = max_abs_err(t_upw, tex)
    err_spg = max_abs_err(t_spg, tex)
    # 控制台输出误差数据
    print(f"\n========= Pe = {Pe_target:.2f} =========")
    print(f"扩散系数 κ = {kappa:.6f}")
    print(f"标准Galerkin 最大误差：{err_gal:.6e}")
    print(f"迎风格式     最大误差：{err_upw:.6e}")
    print(f"SUPG稳定格式 最大误差：{err_spg:.6e}")
    # 绘图
    plt.figure(figsize=(12,7))
    plt.plot(x_gal, tex, 'k-', lw=2.3, label="精确解(Exact)")
    plt.plot(x_gal, t_gal, 'r--o', markersize=6, lw=1.6, label=f"标准 Galerkin ($\\alpha$=0)")
    plt.plot(x_upw, t_upw, 'b--s', markersize=6, lw=1.6, label=f"迎风格式 ($\\alpha$=1)")
    plt.plot(x_spg, t_spg, 'g--^', markersize=6, lw=1.6, label=f"SUPG ($\\alpha$={alpha_spg:.4f})")
    plt.xlabel("空间坐标 $x$", fontsize=12)
    plt.ylabel(r"场变量 $\theta(x)$", fontsize=12)
    plt.title(f"一维对流扩散有限元数值解，单元Peclet数 $Pe={Pe_target}$", fontsize=14)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Pe_{Pe_target}_解对比图.png", dpi=300, bbox_inches="tight")
    plt.show()
    return {"K_gal": K_gal}

def plot_convergence_lenght(Pe_target=3.0, L=1.0, v=1.0):
    """绘制Pe=3下以单元长度le为横轴的双对数收敛误差曲线"""
    nel_list = [5, 10, 20, 40, 80]
    le_list = [L / n for n in nel_list]
    err_gal_list = []
    err_upw_list = []
    err_spg_list = []
    for nel in nel_list:
        le = L / nel
        kappa = v * le / (2 * Pe_target)
        ag = 0.0
        au = 1.0
        aspg = alpha_supg(Pe_target)
        _, tg, tex, _ = solve_advection(nel, L, v, kappa, ag)
        _, tu, _, _ = solve_advection(nel, L, v, kappa, au)
        _, ts, _, _ = solve_advection(nel, L, v, kappa, aspg)
        err_gal_list.append(max_abs_err(tg, tex))
        err_upw_list.append(max_abs_err(tu, tex))
        err_spg_list.append(max_abs_err(ts, tex))
    # 收敛曲线绘图
    plt.figure(figsize=(12,7))
    plt.loglog(le_list, err_gal_list, 'r-o', lw=1.8, markersize=8, label="标准 Galerkin (alpha=0)")
    plt.loglog(le_list, err_upw_list, 'b-s', lw=1.8, markersize=8, label="迎风格式 (alpha=1)")
    plt.loglog(le_list, err_spg_list, 'g-^', lw=1.8, markersize=8, label="SUPG (alpha=alpha_opt)")
    plt.xlabel("单元长度 $l_e$", fontsize=12)
    plt.ylabel("最大节点误差", fontsize=12)
    plt.title("Pe=3.0 对流占优工况 单元长度-误差收敛曲线（双对数坐标）", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("Pe3_单元长度收敛曲线.png", dpi=300, bbox_inches="tight")
    plt.show()
    # 打印收敛数据表
    print("\n========= Pe=3.0 收敛误差数据表 =========")
    print(f"{'单元数nel':<10}{'单元长度le':<14}{'Galerkin误差':<18}{'迎风误差':<18}{'SUPG误差':<18}")
    for i, n in enumerate(nel_list):
        print(f"{n:<10}{le_list[i]:<14.4f}{err_gal_list[i]:<18.6e}{err_upw_list[i]:<18.6e}{err_spg_list[i]:<18.6e}")

def matrix_property_analysis(K):
    """分析Pe=3标准Galerkin刚度矩阵对称性、正定性"""
    print("\n========= Pe=3.0 标准Galerkin刚度矩阵特性 =========")
    is_symmetric = np.allclose(K, K.T)
    print(f"矩阵是否对称：{is_symmetric}")
    eig_vals = np.linalg.eigvalsh(K)
    min_eig = np.min(eig_vals)
    print(f"矩阵最小特征值：{min_eig:.4e}")
    print(f"矩阵是否正定：{min_eig > 0}")
    print(f"前5阶特征值：{eig_vals[:5]}")

if __name__ == "__main__":
    # 1. 绘制Pe=0.1扩散占优解对比图
    res_pe01 = plot_solution_curve(Pe_target=0.1, nel=20)
    # 2. 绘制Pe=3.0对流占优解对比图（会出现Galerkin振荡）
    res_pe30 = plot_solution_curve(Pe_target=3.0, nel=20)
    # 3. 绘制单元长度横轴的收敛误差曲线
    plot_convergence_lenght(Pe_target=3.0)
    # 4. 刚度矩阵性质分析
    matrix_property_analysis(res_pe30["K_gal"])