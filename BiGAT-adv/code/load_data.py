import csv

import dgl
import scipy.sparse as sp
import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralCoclustering, KMeans

from train import train
import numpy as np
  
    
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cd_data = []
        cd_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(cd_data)
    

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)

def load_dataset(args):
    dataset = dict()
    dataset['c_d'] = read_csv(args.dataset_path + '/c_d.csv')
    # dataset['c_d'] = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets2\c_d.csv")
    # dataset['c_d'] = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets\c_d.csv")

    zero_index = []
    one_index = []
    cd_pairs = []
    for i in range(dataset['c_d'].size(0)):
        for j in range(dataset['c_d'].size(1)):
            if dataset['c_d'][i][j] < 1:
                zero_index.append([i, j, 0])
            if dataset['c_d'][i][j] >= 1:
                one_index.append([i, j, 1])

    cd_pairs = random.sample(zero_index, len(one_index)) + one_index

    return dataset, cd_pairs, one_index


def build_dynamic_graph(args, train_cd_pairs):
    # 加载必要的矩阵（不包含全局关联）
    dd_matrix = read_csv(args.dataset_path + '/d_d.csv')
    cc_matrix = read_csv(args.dataset_path + '/c_c.csv')

    # 提取训练集正样本的边
    one_index_train = [[pair[0], pair[1]] for pair in train_cd_pairs if pair[2] == 1]

    # 构建异构图
    dis = np.array(dd_matrix)
    cis = np.array(cc_matrix)
    graph = dgl_heterograph(dis, cis, one_index_train, args)
    return graph, dis, cis

#
# def load_dataset(args):
#     dataset = dict()
#     dataset['c_d'] = read_csv(args.dataset_path + '/c_d.csv')
#     # dataset['c_d'] = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets2\c_d.csv")
#     # dataset['c_d'] = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets\c_d.csv")
#
#     zero_index = []
#     one_index = []
#     cd_pairs = []
#     for i in range(dataset['c_d'].size(0)):
#         for j in range(dataset['c_d'].size(1)):
#             if dataset['c_d'][i][j] < 1:
#                 zero_index.append([i, j, 0])
#             if dataset['c_d'][i][j] >= 1:
#                 one_index.append([i, j, 1])
#
#     cd_pairs = random.sample(zero_index, len(one_index)) + one_index
#
#     dd_matrix = read_csv(args.dataset_path + '/d_d.csv')
#     # dd_matrix = read_csv(args.dataset_path + '/GDD_similarity.csv')
#     # dd_matrix = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets\d_d.csv")
#     dd_edge_index = get_edge_index(dd_matrix)
#     dataset['dd'] = {'data_matrix': dd_matrix, 'edges': dd_edge_index}
#     dis = np.array(dd_matrix)
#
#     # cc_matrix = read_csv(args.dataset_path + '/GCC_similarity.csv')
#     cc_matrix = read_csv(args.dataset_path + '/c_c.csv')
#     # cc_matrix = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets2\c_c.csv")
#     cc_edge_index = get_edge_index(cc_matrix)
#     dataset['cc'] = {'data_matrix': cc_matrix, 'edges': cc_edge_index}
#     cis = np.array(cc_matrix)
#
#     cd_matrix = read_csv(args.dataset_path + '/c_d.csv')
#     cd_edge_index = get_edge_index(cd_matrix)
#     dataset['cd'] = {'data_matrix': cd_matrix, 'edges': cd_edge_index}
#
#     return dataset, cd_pairs, cc_edge_index, dd_edge_index, one_index, cd_edge_index, dis, cis


def dataset(args):
    dataset = dict()

    dataset['c_d'] = read_csv(args.dataset_path + '/c_d.csv')
    zero_index = []
    one_index = []
    cd_pairs = []
    for i in range(dataset['c_d'].size(0)):
        for j in range(dataset['c_d'].size(1)):
            if dataset['c_d'][i][j] < 1:
                zero_index.append([i, j, 0])
            if dataset['c_d'][i][j] >= 1:
                one_index.append([i, j, 1])

    cd_pairs = one_index + random.sample(zero_index, len(one_index))
    cd_pairs = np.array(cd_pairs)  # 转换为 NumPy 数组，方便索引

    # 提取 rating_pairs 和 rating_values
    rating_pairs = (cd_pairs[:, 0], cd_pairs[:, 1])  # (cell 索引, disease 索引)
    rating_values = cd_pairs[:, 2]  # 评分值 (0 或 1)
    # print("Number of 0s:", np.sum(rating_values == 0))
    # print("Number of 1s:", np.sum(rating_values == 1))

    # 读取其他数据
    dd_matrix = read_csv(args.dataset_path + '/d_d.csv')
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd'] = {'data_matrix': dd_matrix, 'edges': dd_edge_index}

    cc_matrix = read_csv(args.dataset_path + '/c_c.csv')
    cc_edge_index = get_edge_index(cc_matrix)
    dataset['cc'] = {'data_matrix': cc_matrix, 'edges': cc_edge_index}

    return dataset, cd_pairs,rating_pairs, rating_values

class GraphConstructor:
    def __init__(self, num_cell, num_disease, possible_rating_values,rna_sim,disease_sim):
        self._num_rna = num_cell
        self._num_disease = num_disease
        self.possible_rating_values = possible_rating_values
        self.rna_sim = rna_sim
        self.disease_sim = disease_sim

    def _generate_enc_graph_adv(self, rating_pairs, rating_values, add_support=False):
        rna_disease_R = np.zeros((self._num_rna, self._num_disease), dtype=np.float32)
        rna_disease_R[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {"rna": self._num_rna, "disease": self._num_disease}
        rating_row, rating_col = rating_pairs

        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = str(rating)
            data_dict.update({
                ("disease", rating, "rna"): (rcol, rrow),
                ("rna", "rev-" + rating, "disease"): (rrow, rcol),
            })
        P_pairs = np.stack([rating_row, rating_col], axis=1)
        all_pairs = np.array([[i, j] for i in range(self._num_rna) for j in range(self._num_disease)])
        R_mask = np.zeros((self._num_rna, self._num_disease), dtype=bool)
        R_mask[rating_row, rating_col] = True
        U_pairs = all_pairs[~R_mask.flatten()]  # 找出没出现的 i,j 对

        # 从上面提到的函数 select_reliable_negatives 获取可靠负样本
        reliable_neg_pairs,reliable_scores = self.select_reliable_negatives(
            P_pairs=P_pairs,
            U_pairs=U_pairs,
            S_r=self.rna_sim,  # 提前准备的 RNA 相似度矩阵
            S_d=self.disease_sim,  # 提前准备的 Disease 相似度矩阵
            max_iter=5,
            tol=1e-3
        )

        # 添加单向“低关联”边（负样本）：
        neg_i, neg_j = reliable_neg_pairs[:, 0], reliable_neg_pairs[:, 1]
        data_dict.update({
            ("disease", "pred_low", "rna"): (neg_j, neg_i)
        })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        scores = torch.tensor(reliable_scores, dtype=torch.float32)
        # 将它赋给 pred_low 边的 'weight'
        graph.edges['pred_low'].data['weight'] = scores
        # graph_homo = dgl.to_homogeneous(graph)
        # heterograph_dict = dgl.to_bidirected(graph_homo)
        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype("float32")
                x[x == 0.0] = np.inf
                x = torch.FloatTensor(1.0 / np.sqrt(x))
                return x.unsqueeze(1)

            gene_ci, gene_cj = [], []
            cell_ci, cell_cj = [], []

            for r in self.possible_rating_values:
                r = str(r)
                gene_ci.append(graph["rev-" + r].in_degrees())
                cell_ci.append(graph[r].in_degrees())
                gene_cj.append(graph[r].out_degrees())
                cell_cj.append(graph["rev-" + r].out_degrees())

            gene_ci = _calc_norm(sum(gene_ci))
            cell_ci = _calc_norm(sum(cell_ci))
            gene_cj = _calc_norm(sum(gene_cj))
            cell_cj = _calc_norm(sum(cell_cj))

            graph.nodes["gene"].data.update({"ci": gene_ci, "cj": gene_cj})

        return graph

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        rna_disease_R = np.zeros((self._num_rna, self._num_disease), dtype=np.float32)
        rna_disease_R[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {"rna": self._num_rna, "disease": self._num_disease}
        rating_row, rating_col = rating_pairs

        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = str(rating)
            data_dict.update({
                ("disease", rating, "rna"): (rcol, rrow),
                ("rna", "rev-" + rating, "disease"): (rrow, rcol),
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        # graph_homo = dgl.to_homogeneous(graph)
        # heterograph_dict = dgl.to_bidirected(graph_homo)
        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype("float32")
                x[x == 0.0] = np.inf
                x = torch.FloatTensor(1.0 / np.sqrt(x))
                return x.unsqueeze(1)

            gene_ci, gene_cj = [], []
            cell_ci, cell_cj = [], []

            for r in self.possible_rating_values:
                r = str(r)
                gene_ci.append(graph["rev-" + r].in_degrees())
                cell_ci.append(graph[r].in_degrees())
                gene_cj.append(graph[r].out_degrees())
                cell_cj.append(graph["rev-" + r].out_degrees())

            gene_ci = _calc_norm(sum(gene_ci))
            cell_ci = _calc_norm(sum(cell_ci))
            gene_cj = _calc_norm(sum(gene_cj))
            cell_cj = _calc_norm(sum(cell_cj))

            graph.nodes["gene"].data.update({"ci": gene_ci, "cj": gene_cj})

        return graph

    '''def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        rna_disease_R = np.zeros((self._num_rna, self._num_disease), dtype=np.float32)
        rna_disease_R[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {"rna": self._num_rna, "disease": self._num_disease}
        rating_row, rating_col = rating_pairs

        # 只保留从 rna 到 disease 的边，单向边
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = str(rating)
            data_dict.update({
                # ("rna", rating, "disease"): (rrow, rcol),
                # ("rna", rating, "disease"): (rrow, rcol),
                # 单向图，不添加反向边
                ("disease", rating, "rna"): (rcol, rrow),
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype("float32")
                x[x == 0.0] = np.inf
                x = torch.FloatTensor(1.0 / np.sqrt(x))
                return x.unsqueeze(1)

            # 对于单向边，只计算正向边的信息：
            # 对于 "rna" 节点，使用其作为源节点的出度；
            # 对于 "disease" 节点，使用其作为目标节点的入度。
            rna_deg_sum = 0
            dis_deg_sum = 0
            for rating in self.possible_rating_values:
                rating = str(rating)
                rna_deg_sum += graph.out_degrees(etype=rating)
                dis_deg_sum += graph.in_degrees(etype=rating)
            rna_norm = _calc_norm(rna_deg_sum)
            dis_norm = _calc_norm(dis_deg_sum)

            # 这里将支持信息更新到各自节点上
            graph.nodes["rna"].data.update({"ci": rna_norm, "cj": rna_norm})
            graph.nodes["disease"].data.update({"ci": dis_norm, "cj": dis_norm})

        return graph'''



    def _generate_dec_graph(self, rating_pairs):


        ones = np.ones_like(rating_pairs[0])
        gene_cell_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self._num_rna, self._num_disease),
            dtype=np.float32,
        )
        g = dgl.bipartite_from_scipy(gene_cell_ratings_coo, utype="_U", etype="_E", vtype="_V")
        return dgl.heterograph(
            {("rna", "rate", "disease"): g.edges()},
            num_nodes_dict={"rna": self._num_rna, "disease": self._num_disease},
        )

    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import torch

    def select_reliable_negatives(
            self,
            P_pairs,  # 正样本索引对  shape [|P|,2]
            U_pairs,  # 候选负样本对  shape [|U|,2]
            S_r, S_d,  # RNA 和疾病相似度矩阵
            max_iter=10,
            tol=1e-3
    ):
        """
        MG-CNSS 算法：先用 KMeans 初始化质心，然后交替余弦＋欧式相似度划分，
        直到质心收敛，输出可靠负样本对及其 ES 得分。
        """
        # 1) 构造特征向量 Fp, Fu
        Fr, Fd = S_r, S_d  # [Nr×Nr], [Nd×Nd]

        def make_F(pairs):
            return np.hstack([Fr[pairs[:, 0], :], Fd[pairs[:, 1], :]])

        Fp = make_F(P_pairs)  # [|P|, Nr+Nd]
        Fu = make_F(U_pairs)  # [|U|, Nr+Nd]

        # 2) 用 KMeans 对 未标记对 Fu 做 2 类聚类，初始化 Cu
        kmeans = KMeans(n_clusters=2, random_state=0).fit(Fu)
        centers = kmeans.cluster_centers_  # shape [2, Nr+Nd]

        # 3) 正样本质心 Cp 直接用 P 的均值
        Cp = Fp.mean(axis=0)

        # 4) 选出哪一类中心更像负样本——与 Cp 距离更远
        d0 = np.linalg.norm(centers[0] - Cp)
        d1 = np.linalg.norm(centers[1] - Cp)
        Cu = centers[0] if d0 > d1 else centers[1]

        # 5) 迭代余弦（CS）+欧氏相似度（ES）划分
        for _ in range(max_iter):
            Cp_prev, Cu_prev = Cp.copy(), Cu.copy()

            # 5.1) 计算余弦相似度划分 P₁ vs N₁
            cs_p = cosine_similarity(Fu, Cp[None, :]).ravel()
            cs_u = cosine_similarity(Fu, Cu[None, :]).ravel()
            idx_pos = np.where(cs_p > cs_u)[0]
            idx_neg = np.where(cs_p <= cs_u)[0]

            # 5.2) 根据 P ∪ P₁ 更新 Cp，根据 N₁ 更新 Cu
            Cp = np.vstack([Fp, Fu[idx_pos]]).mean(axis=0)
            Cu = Fu[idx_neg].mean(axis=0)

            # 5.3) 计算欧氏相似度 ESₚ, ESᵤ
            dist_p = np.sum((Fu - Cp) ** 2, axis=1)
            dist_u = np.sum((Fu - Cu) ** 2, axis=1)
            es_p = 1.0 / (1.0 + dist_p)
            es_u = 1.0 / (1.0 + dist_u)

            # 5.4) 重新划分：P′₁ vs N′₁
            idx_pos2 = np.where(es_p > es_u)[0]
            idx_neg2 = np.where(es_p <= es_u)[0]

            # 5.5) 更新质心
            Cp = np.vstack([Fp, Fu[idx_pos2]]).mean(axis=0)
            Cu = Fu[idx_neg2].mean(axis=0)

            # 5.6) 收敛判断
            if np.linalg.norm(Cp - Cp_prev) < tol and np.linalg.norm(Cu - Cu_prev) < tol:
                break

        # 6) 返回可靠负样本对及其 ESᵤ 分数作为 weight
        reliable_neg_pairs = U_pairs[idx_neg2]
        reliable_scores = es_u[idx_neg2]

        return reliable_neg_pairs, reliable_scores

    '''def select_reliable_negatives(
            self,P_pairs, U_pairs, S_r, S_d,
            max_iter=10, tol=1e-3
    ):
        # 1) 构造特征向量 F for all P+U
        #    这里我用“相似度矩阵的行”作为特征，也可以先降维/嵌入再拼接
        Fr = S_r  # shape [Nr, Nr]
        Fd = S_d  # shape [Nd, Nd]

        def make_F(pairs):
            # pairs: array [[i1,j1],[i2,j2],...]
            return np.hstack([
                Fr[pairs[:, 0], :],  # RNA_i row
                Fd[pairs[:, 1], :]  # Disease_j row
            ])  # shape [num_pairs, Nr+Nd]

        Fp = make_F(P_pairs)  # 正样本特征
        Fu = make_F(U_pairs)  # 未标记样本特征

        # 2) 初始化质心
        Cp = Fp.mean(axis=0)
        Cu = Fu.mean(axis=0)

        for it in range(max_iter):
            Cp_prev, Cu_prev = Cp.copy(), Cu.copy()

            # 3.1) 余弦相似度划分
            #    cos(Fu, Cp) 与 cos(Fu, Cu)
            cs_p = cosine_similarity(Fu, Cp[None, :]).ravel()
            cs_u = cosine_similarity(Fu, Cu[None, :]).ravel()
            idx_pos = np.where(cs_p > cs_u)[0]
            idx_neg = np.where(cs_p <= cs_u)[0]

            # 3.2) 重新计算质心
            Cp = np.vstack([Fp, Fu[idx_pos]]).mean(axis=0)  # 把已知正样本也继续保留
            Cu = Fu[idx_neg].mean(axis=0)

            # 3.3) 细化：欧氏相似度
            #    ES_p = 1/(1+||Fu - Cp||^2), ES_u = 1/(1+||Fu - Cu||^2)
            dist_p = np.sum((Fu - Cp) ** 2, axis=1)
            dist_u = np.sum((Fu - Cu) ** 2, axis=1)
            es_p = 1 / (1 + dist_p)
            es_u = 1 / (1 + dist_u)
            idx_pos2 = np.where(es_p > es_u)[0]
            idx_neg2 = np.where(es_p <= es_u)[0]

            # 用细化结果更新 neg set
            idx_neg = idx_neg2

            # 最终更新 Cp', Cu'
            Cp = np.vstack([Fp, Fu[idx_pos2]]).mean(axis=0)
            Cu = Fu[idx_neg2].mean(axis=0)

            # 检查收敛
            if np.linalg.norm(Cp - Cp_prev) < tol and np.linalg.norm(Cu - Cu_prev) < tol:
                break

        # 返回可靠负样本对
        reliable_neg_pairs = U_pairs[idx_neg2]
        reliable_scores = es_u[idx_neg2]
        return reliable_neg_pairs,reliable_scores'''

def feature_representation(model, args, dataset,enc_graph,enc_graph_adv,dec_graph,rating_values):
    model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model = train(model, dataset, enc_graph, enc_graph_adv, dec_graph, optimizer, rating_values, args, ufeat=None,
                  ifeat=None)
    model.eval()
    with torch.no_grad():
        score, cir_fea, dis_fea = model(args,enc_graph,dec_graph,None,None,dataset)
    cir_fea = cir_fea.cpu().detach().numpy()
    dis_fea = dis_fea.cpu().detach().numpy()
    return score, cir_fea, dis_fea

'''def feature_representation(model, args, dataset,enc_graph,enc_graph_adv,dec_graph,rating_values):
    model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model = train(model, dataset,enc_graph,enc_graph_adv,dec_graph, optimizer,rating_values, args,ufeat=None,ifeat=None)
    torch.save(model.state_dict(), 'pretrained.pth')
    model = finetune(
            model, dataset,
            enc_graph, enc_graph_adv, dec_graph,
            rating_values, opt,
            ufeat, ifeat,
            lambda_adv=1.5, epsilon=0.001,
            finetune_epochs=10,
            lr=1e-3, weight_decay=1e-4,
            pretrained_path='pretrained.pth'
        )
    model.eval()
    with torch.no_grad():
        score, cir_fea, dis_fea = model(args,enc_graph, dec_graph,None,None,dataset)
    cir_fea = cir_fea.cpu().detach().numpy()
    dis_fea = dis_fea.cpu().detach().numpy()
    return score, cir_fea, dis_fea'''

'''def feature_representation(
    model,
    args,
    dataset,
    enc_graph,        # 只包含双向真实关联的图
    enc_graph_adv,    # 包含 pred_low 单向边的对抗图
    dec_graph,
    rating_values,
    pretrain_epochs=50,
    finetune_epochs=10,
    lr_pre=1e-3,
    lr_ft=1e-3,
    weight_decay=1e-4,
):
    # ——— 1) 预训练阶段 ———
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_pre, weight_decay=weight_decay)
    # train() 内部默认使用 enc_graph 和 dec_graph 计算 loss_standard + λ*loss_adv，
    # 但我们这里暂时把 λ 设为 0，只进行标准训练即可：
    model.train()
    for epoch in range(pretrain_epochs):
        score, _, _ = model(args, enc_graph, dec_graph, None, None, dataset)
        loss = torch.nn.MSELoss()(
            torch.sigmoid(score).squeeze(),
            torch.tensor(rating_values, dtype=torch.float32, device=score.device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"[Pretrain] Epoch {epoch}/{pretrain_epochs}, Loss: {loss.item():.4f}")
    # 保存预训练好的参数
    torch.save(model.state_dict(), "pretrained.pth")

    # ——— 2) 对抗微调阶段 ———
    # 加载预训练权重
    model.load_state_dict(torch.load("pretrained.pth"))

    # 冻结所有参数
    for name, p in model.named_parameters():
        p.requires_grad = False
    # 只解冻 pred_low 边的 α（或 gate_scalar）和 ufc_adv
    for name, p in model.named_parameters():
        if "alpha" in name or "ufc_adv" in name:
            p.requires_grad = True

    # 只对解冻参数建优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_ft, weight_decay=weight_decay
    )

    model.train()
    for epoch in range(finetune_epochs):
        # 标准图损失（λ=1.0）
        score_std, _, _ = model(args, enc_graph, dec_graph, None, None, dataset)
        loss_std = torch.nn.MSELoss()(
            torch.sigmoid(score_std).squeeze(),
            torch.tensor(rating_values,dtype=torch.float32, device=score_std.device)
        )
        # 对抗图损失
        score_adv, _, _ = model(args, enc_graph_adv, dec_graph, None, None, dataset)
        loss_adv = torch.nn.MSELoss()(
            torch.sigmoid(score_adv).squeeze(),
            torch.tensor(rating_values,dtype=torch.float32, device=score_adv.device)
        )
        loss = loss_std +1.5 * loss_adv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"[Finetune] Epoch {epoch}/{finetune_epochs}, Loss: {loss.item():.4f}")

    # ——— 3) 特征提取 ———
    model.eval()
    with torch.no_grad():
        score, cir_fea, dis_fea = model(args, enc_graph, dec_graph, None, None, dataset)
    # 转 numpy 返回
    return (
        score.cpu().numpy(),
        cir_fea.cpu().numpy(),
        dis_fea.cpu().numpy()
    )'''


def new_dataset(cir_fea, dis_fea, cd_pairs):
    unknown_pairs = []
    known_pairs = []
    
    for pair in cd_pairs:
        if pair[2] == 1:
            known_pairs.append(pair[:2])
            
        if pair[2] == 0:
            unknown_pairs.append(pair[:2])
    # print("--------------------")
    # print(cir_fea.shape,dis_fea.shape)
    # print("--------------------")
    # print(len(unknown_pairs), len(known_pairs))
    #
    nega_list = []
    for i in range(len(unknown_pairs)):
        nega = cir_fea[unknown_pairs[i][0],:].tolist() + dis_fea[unknown_pairs[i][1],:].tolist()+[0,1]
        nega_list.append(nega)
        
    posi_list = []
    for j in range(len(known_pairs)):
        posi = cir_fea[known_pairs[j][0],:].tolist() + dis_fea[known_pairs[j][1],:].tolist()+[1,0]
        posi_list.append(posi)
    
    samples = posi_list + nega_list
    
    random.shuffle(samples)
    samples = np.array(samples)
    return samples

def dgl_heterograph(dis,cis,rdi, args):
    # rdi 是一个元组，包含源节点和目标节点的张量
    src_nodes = torch.tensor([item[0] for item in rdi])
    dst_nodes = torch.tensor([item[1] for item in rdi])
    # 创建节点数量字典
    node_dict = {
        'rna': args.circRNA_number,
        'disease': args.disease_number,
    }
    # 创建异构图字典
    heterograph_dict = {
        ('rna', '1', 'disease'): (src_nodes, dst_nodes),
        ('disease', 'rev-1', 'rna'): (dst_nodes, src_nodes),
    }
    # 创建异构图
    crna_di_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)
    rna_features = torch.tensor(cis) # 3 个 RNA 节点，每个节点 5 个特征
    disease_features = torch.tensor(dis) # 3 个 Disease 节点，每个节点 5 个特征
    crna_di_graph.nodes['rna'].data['features'] = rna_features
    crna_di_graph.nodes['disease'].data['features'] = disease_features
    return crna_di_graph


# def C_Dmatix(cd_pairs,trainindex,testindex):
#     c_dmatix = np.zeros((585,88))
#     for i in trainindex:
#         if cd_pairs[i][2]==1:
#             c_dmatix[cd_pairs[i][0]][cd_pairs[i][1]]=1
#     dataset = dict()
#     cd_data = []
#     cd_data += [[float(i) for i in row] for row in c_dmatix]
#     cd_data = torch.Tensor(cd_data)
#     dataset['c_d'] = cd_data
#     train_cd_pairs = []
#     test_cd_pairs = []
#     for m in trainindex:
#         train_cd_pairs.append(cd_pairs[m])
#
#     for n in testindex:
#         test_cd_pairs.append(cd_pairs[n])
#     return dataset['c_d'],train_cd_pairs,test_cd_pairs


def C_Dmatix(cd_pairs, trainindex, testindex):
    # 仅基于训练集正样本生成关联矩阵
    c_dmatix = np.zeros((585, 88))  # 替换为从args获取实际维度
    for i in trainindex:
        if cd_pairs[i][2] == 1:  # 仅保留训练集正样本
            c_dmatix[cd_pairs[i][0]][cd_pairs[i][1]] = 1

    # 生成新的dataset
    dataset = dict()
    dataset['c_d'] = torch.Tensor(c_dmatix)

    # 分割训练集和测试集
    train_cd_pairs = [cd_pairs[i] for i in trainindex]
    test_cd_pairs = [cd_pairs[i] for i in testindex]

    return dataset['c_d'], train_cd_pairs, test_cd_pairs