import numpy as np
import json

class TrussFEM:
    def __init__(self, json_path):
        self.load_model(json_path)
        self.init_dof()
        self.build_LM()
        self.assemble_global_stiffness()

    def load_model(self, json_path):
        # 读取json模型文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 基础参数
        self.nsd = data["nsd"]
        self.ndof_node = data["ndof"]
        self.nnp = data["nnp"]
        self.nel = data["nel"]
        self.nen = data["nen"]
        # 材料截面
        self.E = np.array(data["E"], dtype=float)
        self.A = np.array(data["CArea"], dtype=float)
        # 节点坐标
        self.x = np.array(data["x"], dtype=float)
        self.y = np.array(data["y"], dtype=float) if self.nsd >= 2 else None
        # 单元连接，转换为0起始索引
        self.IEN = np.array(data["IEN"], dtype=int) - 1
        # 位移边界条件
        self.fixed_dof = np.array(data["fixed_dof"], dtype=int) - 1
        self.fixed_val = np.array(data["fixed_value"], dtype=float)
        # 节点载荷
        self.force_dof = np.array(data["force_dof"], dtype=int) - 1
        self.force_val = np.array(data["force_value"], dtype=float)

    def init_dof(self):
        # 初始化总自由度、位移向量、载荷向量
        self.ndof_total = self.nnp * self.ndof_node
        self.d = np.zeros(self.ndof_total)
        self.f = np.zeros(self.ndof_total)
        for dof, val in zip(self.force_dof, self.force_val):
            self.f[dof] = val

    def build_LM(self):
        # 生成局部-全局自由度映射LM矩阵
        self.LM = np.zeros((self.nen * self.ndof_node, self.nel), dtype=int)
        print("===== LM 对号矩阵（局部自由度→全局自由度） =====")
        for e in range(self.nel):
            n1, n2 = self.IEN[e, 0], self.IEN[e, 1]
            # 节点1自由度映射
            for i in range(self.ndof_node):
                self.LM[i, e] = n1 * self.ndof_node + i
            # 节点2自由度映射
            for i in range(self.ndof_node):
                self.LM[self.ndof_node + i, e] = n2 * self.ndof_node + i
            print(f"单元{e+1} 映射向量：{self.LM[:, e]}")
        print()

    def calc_element_stiffness(self, e):
        # 计算单个单元刚度矩阵
        n1, n2 = self.IEN[e, 0], self.IEN[e, 1]
        E = self.E[e]
        A = self.A[e]
        dx = self.x[n2] - self.x[n1]
        if self.nsd == 1:
            L = dx
            c, s = 1.0, 0.0
            Ke = (E * A / L) * np.array([[1, -1], [-1, 1]])
        else:
            dy = self.y[n2] - self.y[n1]
            L = np.hypot(dx, dy)
            c = dx / L  # x方向余弦
            s = dy / L  # y方向余弦
            Ke = (E * A / L) * np.array([
                [c**2, c*s, -c**2, -c*s],
                [c*s, s**2, -c*s, -s**2],
                [-c**2, -c*s, c**2, c*s],
                [-c*s, -s**2, c*s, s**2]
            ])
        self.elem_L, self.elem_c, self.elem_s = L, c, s
        # 打印当前单元的方向余弦（可选：如果需要在刚度计算时就输出）
        # print(f"单元{e+1} 方向余弦：c={c:.6f}, s={s:.6f}")
        return Ke

    def assemble_global_stiffness(self):
        # 组装总体刚度矩阵
        self.K = np.zeros((self.ndof_total, self.ndof_total))
        print("===== 各单元方向余弦 =====")
        for e in range(self.nel):
            # 先计算单元刚度以获取方向余弦
            self.calc_element_stiffness(e)
            Ke = self.calc_element_stiffness(e)
            lm = self.LM[:, e]
            for a in range(len(lm)):
                for b in range(len(lm)):
                    self.K[lm[a], lm[b]] += Ke[a, b]
            # 输出当前单元的方向余弦
            print(f"单元{e+1}：c={self.elem_c:.6f}, s={self.elem_s:.6f}")
        print()

        # 刚度矩阵性质检查（和截图输出格式完全匹配）
        print("=== 总体刚度矩阵性质检查 ===")
        diff_K = np.abs(self.K - self.K.T)
        max_diff = np.max(diff_K)
        print(f"最大不对称量：{max_diff:.2e}（应为0或极小值）")
        eig_vals = np.linalg.eigvals(self.K)
        zero_eig = np.sum(np.isclose(eig_vals, 0, atol=1e-9))
        print(f"零特征值个数(奇异度)：{zero_eig}（至少应为 1）")
        neg_diag = np.sum(np.diag(self.K) < -1e-9)
        print(f"负对角元个数：{neg_diag}\n")

        print("总体刚度矩阵（前几行几列）：")
        print(np.round(self.K, 6))
        print()

    def apply_bc_reduction(self):
        # 缩减法施加边界条件，求解位移与约束反力
        all_dof = np.arange(self.ndof_total)
        F_dof = np.setdiff1d(all_dof, self.fixed_dof)
        E_dof = self.fixed_dof
        # 刚度分块
        Kff = self.K[np.ix_(F_dof, F_dof)]
        Kfe = self.K[np.ix_(F_dof, E_dof)]
        f_F = self.f[F_dof]
        d_E = np.zeros_like(E_dof, dtype=float)
        for idx, _ in enumerate(E_dof):
            d_E[idx] = self.fixed_val[idx]
        # 求解未知位移
        rhs = f_F - Kfe @ d_E
        d_F = np.linalg.solve(Kff, rhs)
        self.d[F_dof] = d_F
        self.d[E_dof] = d_E
        # 计算约束反力
        Kef = self.K[np.ix_(E_dof, F_dof)]
        Kee = self.K[np.ix_(E_dof, E_dof)]
        f_E = self.f[E_dof]
        self.R = Kef @ d_F + Kee @ d_E - f_E

        # 打印节点位移
        print("=== 节点位移 ===")
        if self.nsd == 1:
            for i in range(self.nnp):
                print(f"节点 {i+1}: 位移 = {self.d[i]:.6f}")
        else:
            for i in range(self.nnp):
                u = self.d[i*2]
                v = self.d[i*2+1]
                print(f"节点 {i+1}: u={u:.6f}, v={v:.6f}")
        print()

        # 打印约束反力
        print("=== 约束反力 ===")
        for idx, dof in enumerate(E_dof):
            node_id = dof // self.ndof_node + 1
            if self.ndof_node == 1:
                dir_name = "x"
            else:
                dir_name = "x" if dof % 2 == 0 else "y"
            print(f"自由度 {dof+1} (节点{node_id},{dir_name}): 反力 = {self.R[idx]:.6f}")
        print()

    def post_process(self):
        # 单元后处理：长度、应力、轴力 + 方向余弦
        print("=== 单元结果 ===\n")
        for e in range(self.nel):
            self.calc_element_stiffness(e)
            lm = self.LM[:, e]
            de = self.d[lm]
            L, c, s = self.elem_L, self.elem_c, self.elem_s
            E, A = self.E[e], self.A[e]
            if self.nsd == 1:
                sigma = (E / L) * np.array([-1, 1]) @ de
            else:
                sigma = (E / L) * np.array([-c, -s, c, s]) @ de
            N = sigma * A
            print(f"单元 {e+1}:")
            print(f"  长度 = {L:.6f}")
            print(f"  方向余弦：c={c:.6f}, s={s:.6f}")  # 新增方向余弦输出
            print(f"  应力 = {sigma:.6f}")
            print(f"  轴力 = {N:.6f}\n")

def main():
    # ========== 切换算例 ==========
    # 算例1 一维两杆（匹配你截图输出）
    fem = TrussFEM("case1_1d_truss.json")
    # 算例2 二维桁架，取消下面注释、注释上面一行即可切换
    # fem = TrussFEM("case2_2d_truss.json")

    fem.apply_bc_reduction()
    fem.post_process()

if __name__ == "__main__":
    main()