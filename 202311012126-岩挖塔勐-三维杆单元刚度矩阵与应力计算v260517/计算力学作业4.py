import numpy as np


def truss3d_element(node1, node2, E, A):
    """三维杆单元：计算长度、方向余弦、6×6刚度矩阵"""
    x1, y1, z1 = node1
    x2, y2, z2 = node2

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    if L < 1e-12:
        raise ValueError("错误：两个节点重合，单元长度为0，属于退化单元！")

    cx = dx / L
    cy = dy / L
    cz = dz / L

    k = E * A / L
    Ke = k * np.array([
        [cx * cx, cx * cy, cx * cz, -cx * cx, -cx * cy, -cx * cz],
        [cx * cy, cy * cy, cy * cz, -cx * cy, -cy * cy, -cy * cz],
        [cx * cz, cy * cz, cz * cz, -cx * cz, -cy * cz, -cz * cz],
        [-cx * cx, -cx * cy, -cx * cz, cx * cx, cx * cy, cx * cz],
        [-cx * cy, -cy * cy, -cy * cz, cx * cy, cy * cy, cy * cz],
        [-cx * cz, -cy * cz, -cz * cz, cx * cz, cy * cz, cz * cz]
    ])

    return L, (cx, cy, cz), Ke


def truss3d_stress_strain(node1, node2, E, A, de):
    """根据节点位移计算应变、应力、轴力"""
    x1, y1, z1 = node1
    x2, y2, z2 = node2
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if L < 1e-12:
        raise ValueError("错误：单元长度为0，无法计算应力应变！")

    cx = dx / L
    cy = dy / L
    cz = dz / L

    B = (1 / L) * np.array([-cx, -cy, -cz, cx, cy, cz])

    epsilon = B @ de
    sigma = E * epsilon
    N = sigma * A

    return epsilon, sigma, N


# ===================== 主程序：严格匹配算例1和算例2 =====================
if __name__ == "__main__":
    # 强制完整输出矩阵，不截断、不省略
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 60)
    print("算例1：沿x轴的一维杆单元")
    print("=" * 60)

    # 算例1参数
    node1_1 = (0, 0, 0)
    node2_1 = (2, 0, 0)
    E1 = 200e9  # 200 GPa
    A1 = 1.0e-4  # 1.0e-4 m²
    de1 = np.array([0, 0, 0, 1.0e-3, 0, 0])  # 位移向量

    L1, (cx1, cy1, cz1), Ke1 = truss3d_element(node1_1, node2_1, E1, A1)
    eps1, sig1, N1 = truss3d_stress_strain(node1_1, node2_1, E1, A1, de1)

    print("节点 1:", node1_1)
    print("节点 2:", node2_1)
    print(f"E = {E1 / 1e9:.0f} GPa")
    print(f"A = {A1:.1e} m^2")
    print(f"de = {de1} m\n")

    print("检查要求结果：")
    print(f"1. 单元长度: L = {L1:.0f} m (应为 2 m)")
    print(f"2. 方向余弦: (cx, cy, cz) = ({cx1:.0f}, {cy1:.0f}, {cz1:.0f}) (应为 (1, 0, 0))")
    print("3. 刚度矩阵 Ke:")
    print(Ke1)
    print(f"4. 轴向应变: ε = {eps1:.1e} (应为 5.0e-4)")
    print(f"5. 轴向应力: σ = {sig1 / 1e6:.0f} MPa (应为 100 MPa)")
    print(f"6. 轴力: N = {N1:.0e} N (应为 1.0e4 N)")

    print("\n" + "=" * 60)
    print("算例2：空间任意方向杆单元")
    print("=" * 60)

    # 算例2参数
    node1_2 = (0, 0, 0)
    node2_2 = (1, 2, 2)
    E2 = 210e9  # 210 GPa
    A2 = 2.0e-4  # 2.0e-4 m²
    de2 = np.array([0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3])  # 位移向量

    L2, (cx2, cy2, cz2), Ke2 = truss3d_element(node1_2, node2_2, E2, A2)
    eps2, sig2, N2 = truss3d_stress_strain(node1_2, node2_2, E2, A2, de2)

    print("节点 1:", node1_2)
    print("节点 2:", node2_2)
    print(f"E = {E2 / 1e9:.0f} GPa")
    print(f"A = {A2:.1e} m^2")
    print(f"de = {de2} m\n")

    print("检查要求结果：")
    print(f"1. 单元长度: L = {L2:.0f} m (应为 3 m)")
    print(f"2. 方向余弦: (cx, cy, cz) = ({cx2:.4f}, {cy2:.4f}, {cz2:.4f}) (应为 (1/3, 2/3, 2/3))")

    # ========== 我只加了这一行，输出算例2完整矩阵 ==========
    print("3. 刚度矩阵 Ke:")
    print(Ke2)
    # =====================================================

    print("4. 刚度矩阵 Ke 是否对称:", np.allclose(Ke2, Ke2.T))
    print("5. 刚体平移测试：位移 [1,1,1,2,2,2]")
    de_rigid = np.array([1, 1, 1, 2, 2, 2])
    eps_rigid, _, _ = truss3d_stress_strain(node1_2, node2_2, E2, A2, de_rigid)
    print(f"   对应应变: ε = {eps_rigid:.2e} (理论上应为0)")
    print("6. Ke 特征值非负性检查：")
    eigvals = np.linalg.eigvalsh(Ke2)
    print(f"   特征值: {np.round(eigvals, 2)}")
    print("   单个自由杆单元刚度矩阵奇异原因：存在刚体位移模式（平动和转动），导致行列式为0，不可逆。\n")
    print(f"7. 轴向应变: ε = {eps2:.1e} (应为 1.0e-3)")
    print(f"8. 轴向应力: σ = {sig2 / 1e6:.0f} MPa (应为 210 MPa)")
    print(f"9. 轴力: N = {N2:.1e} N (应为 4.2e4 N)")