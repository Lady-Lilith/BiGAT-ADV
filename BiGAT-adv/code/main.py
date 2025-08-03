import os

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from param import parameter_parser
import load_data
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import evaluation_scores
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
import time
from mymodel import DGLLayer, Decoder, LinearNet, DGLLayerHomogeneous, read_csv
from torch import nn
args = parameter_parser()
def plot_combined_curves(roc_data, pr_data):
    # 设置图形样式
    plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({'font.size': 14, 'figure.figsize': (16, 7)})

    # 计算平均ROC曲线
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for (fpr, tpr, _) in roc_data:
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)

    # 计算平均PR曲线
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    for (precision, recall, _) in pr_data:
        # 确保按recall排序
        recall, precision = zip(*sorted(zip(recall, precision)))
        recall = np.array(recall)
        precision = np.array(precision)
        interp_precision = np.interp(mean_recall, recall, precision)
        precisions.append(interp_precision)
    mean_precision = np.mean(precisions, axis=0)
    mean_auprc = np.trapz(mean_precision, mean_recall)
    std_precision = np.std(precisions, axis=0)

    # 创建子图
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    # ROC子图：绘制每一折和平均ROC曲线（无阴影）
    for i, (fpr, tpr, roc_auc) in enumerate(roc_data):
        ax[0].plot(fpr, tpr, lw=1.5, alpha=0.7,
                   label=f'Fold {i + 1} ROC (AUC = {roc_auc:.3f})')
    ax[0].plot(mean_fpr, mean_tpr, color='navy', lw=3,
               label=f'Mean ROC (AUC = {mean_auc:.3f})')
    ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC Curve')
    ax[0].legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

    # PR子图：绘制每一折和平均PR曲线（无阴影）
    for i, (precision, recall, pr_auc) in enumerate(pr_data):
        ax[1].plot(recall, precision, lw=1.5, alpha=0.7,
                   label=f'Fold {i + 1} PR (AUPRC = {pr_auc:.3f})')
    ax[1].plot(mean_recall, mean_precision, color='darkred', lw=3,
               label=f'Mean PR (AUPRC = {mean_auprc:.3f})')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].legend(loc='lower left', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self._act = nn.ReLU()
        self.encoder = DGLLayer(
            args,
        [1],
       args.circRNA_number,
        args.disease_number,
        args.msg,
       args.out_channels,
        0,

    )
    #     self.encoder = DGLLayerHomogeneous(
    #             args,
    #             585,  # gene_in_units（RNA 特征维度）
    #             88,  # cell_in_units（disease 特征维度）
    #             512,  # msg_units，同构卷积输出的总维度
    #             256,  # out_units，后续全连接层输出维度
    #             0  # dropout_rate
    #         )
        self.args = args
        self.decoder = Decoder(args,dropout_rate=0)
        self.transformer = LinearNet(args,emb_dim=args.out_channels, num_heads=4, num_layers=2, dropout_rate=0.5)
        self.lc = nn.Linear(self.args.circRNA_number,args.out_channels)
        self.ld = nn.Linear(self.args.disease_number,args.out_channels)
        self.alpha_layer = nn.Linear(args.out_channels, args.out_channels)


    def forward(self, args,enc_graph,dec_graph, ufeat, ifeat,dataset):
        score,cir_fea, dis_fea = self.encoder(args,enc_graph, ufeat, ifeat)
        # score,cir_fea, dis_fea = self.encoder(enc_graph, ufeat, ifeat)
        pred_ratings,cir_fea,dis_fea = self.decoder(dec_graph, cir_fea, dis_fea)
        c_c = self.lc(torch.tensor(dataset['cc']['data_matrix']))
        d_d = self.ld(torch.tensor(dataset['dd']['data_matrix']))
        cir_ = self.transformer(c_c)
        dis_ = self.transformer(d_d)
        alphacir = torch.sigmoid(self.alpha_layer(cir_+cir_fea))
        alphadis = torch.sigmoid(self.alpha_layer(dis_+dis_fea))
        cir = alphacir*cir_fea + (1-alphacir)*cir_
        dis = alphadis*dis_fea + (1-alphadis)*dis_
        return pred_ratings, cir, dis

def save_features_to_csv(cir_fea, dis_fea, cir_filename='cir_fea.csv', dis_filename='dis_fea.csv'):
    # 如果是 PyTorch 张量，先转为 NumPy
    os.makedirs(os.path.dirname(cir_filename), exist_ok=True)
    os.makedirs(os.path.dirname(dis_filename), exist_ok=True)

    if isinstance(cir_fea, torch.Tensor):
        cir_fea = cir_fea.detach().cpu().numpy()
    if isinstance(dis_fea, torch.Tensor):
        dis_fea = dis_fea.detach().cpu().numpy()

    # 保存为 CSV
    pd.DataFrame(cir_fea).to_csv(cir_filename, index=False)
    pd.DataFrame(dis_fea).to_csv(dis_filename, index=False)


def CDA(n_fold):
    args = parameter_parser()
    dataset, cd_pairs, rating_pairs, rating_values= load_data.dataset(args)
 
    
    kf = KFold(n_splits = n_fold, shuffle = True)

    model =Net(args=args)


    ave_acc = 0
    ave_prec = 0
    ave_sens = 0
    ave_f1_score = 0
    ave_mcc = 0
    ave_auc = 0
    ave_auprc = 0
    # 存储各折的ROC/PR曲线数据
    fold_roc_data = []
    fold_pr_data = []

    # 用于记录所有折合并后的真实标签与预测概率
    overall_y_test_list = []
    overall_y_prob_list = []
    localtime = time.asctime( time.localtime(time.time()) )
    with open('../results/GraphCDA-100--5fold.txt', 'a') as f:
        f.write('time:\t'+ str(localtime)+"\n")
        fold_idx = 1
        
        for train_index, test_index in kf.split(cd_pairs):
            c_dmatix,train_cd_pairs,test_cd_pairs = load_data.C_Dmatix(cd_pairs,train_index,test_index)
            dataset['c_d']=c_dmatix
            num_rna = dataset['c_d'].size(0)  # rna数
            num_disease = dataset['c_d'].size(1)  # 疾病数
            possible_rating_values = np.array([1])  # 可能的评分值
            rna_sim = read_csv(args.dataset_path + '/c_c.csv')
            disease_sim = read_csv(args.dataset_path + '/d_d.csv')
            graph_constructor = load_data.GraphConstructor(num_rna, num_disease, possible_rating_values,rna_sim, disease_sim)
            enc_graph_adv = graph_constructor._generate_enc_graph_adv(rating_pairs, rating_values)
            enc_graph = graph_constructor._generate_enc_graph(rating_pairs, rating_values)
            dec_graph = graph_constructor._generate_dec_graph(rating_pairs)
            # crna_di_graph, dis, cis = load_data.build_dynamic_graph(args, train_cd_pairs)
            # 将图传递到特征提取函数
            score, cir_fea, dis_fea = load_data.feature_representation(model, args, dataset, enc_graph,enc_graph_adv,dec_graph,rating_values)
            save_features_to_csv(cir_fea, dis_fea,
                                 cir_filename='./results/cir_fea.csv',
                                 dis_filename='./results/dis_fea.csv')

            train_dataset = load_data.new_dataset(cir_fea, dis_fea, train_cd_pairs)
            test_dataset = load_data.new_dataset(cir_fea, dis_fea, test_cd_pairs)
            # X_train, y_train = train_dataset[:,:-2], train_dataset[:,-2:]
            # X_test, y_test = test_dataset[:,:-2], test_dataset[:,-2:]
            # 修改后（正确）：
            X_train = train_dataset[:, :-2]
            y_train = train_dataset[:, -2]  # 取第一个元素作为标签（1或0）
            X_test = test_dataset[:, :-2]
            y_test = test_dataset[:, -2]
            # 定义随KNN分类器  参数N为邻居数目
            # clf = KNeighborsClassifier(n_neighbors=10)

            # clf = AdaBoostClassifier(n_estimators=200, random_state=42)
            # print(X_train.shape,X_test.shape)
            # clf = RandomForestClassifier(n_estimators=200,n_jobs=11,max_depth=20)
            # scaler = StandardScaler()
            # X_train = scaler.fit_transform(train_dataset[:, :-2])  # 特征部分
            # X_test = scaler.transform(test_dataset[:, :-2])
            clf = GradientBoostingClassifier(n_estimators=200, max_depth=7, random_state=42)
            # clf = XGBClassifier(n_estimators=500, eta=0.1, max_depth=None)
            # clf = MLPClassifier(hidden_layer_sizes=(100, 50),
            #                                         activation='relu',
            #                                         solver='adam',
            #                                         alpha=0.0001,
            #                                         batch_size='auto',
            #                                         learning_rate='constant',
            #                                         learning_rate_init=0.001,
            #                                         max_iter=200,
            #                                         random_state=42,
            #                                         tol=1e-4)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # y_pred = y_pred[:,0]
            y_pred = y_pred
            y_prob = clf.predict_proba(X_test)
            # y_prob = y_prob[1][:,0]
            y_prob = y_prob[:, 1]
            # main.py中修改以下代码：
            # y_prob = clf.predict_proba(X_test)
            # y_prob = np.array([prob[1] for prob in y_prob])  # 提取正类概率
            # tp, fp, tn, fn, acc, prec, sens, f1_score, MCC,AUC,AUPRC = evaluation_scores.calculate_performace(len(y_pred), y_pred, y_prob, y_test[:,0])

            # 将本折的真实标签和预测概率存入overall列表
            overall_y_test_list.extend(list(y_test))
            overall_y_prob_list.extend(list(y_prob))

            tp, fp, tn, fn, acc, prec, sens, f1_score, MCC,AUC,AUPRC = evaluation_scores.calculate_performace(len(y_pred), y_pred, y_prob, y_test)
            print('RF: \n  Acc = \t', acc, '\n  prec = \t', prec, '\n  sens = \t', sens, '\n  f1_score = \t', f1_score, '\n  MCC = \t', MCC, '\n  AUC = \t', AUC,'\n  AUPRC = \t', AUPRC)

            f.write('RF: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(acc)+'\t  prec = \t'+ str(prec)+ '\t  sens = \t'+str(sens)+'\t  f1_score = \t'+str(f1_score)+ '\t  MCC = \t'+str(MCC)+'\t  AUC = \t'+ str(AUC)+'\t  AUPRC = \t'+ str(AUPRC)+'\n')
            # 记录本折的ROC曲线数据
            ave_acc += acc
            ave_prec += prec
            ave_sens += sens
            ave_f1_score += f1_score
            ave_mcc += MCC
            ave_auc += AUC
            ave_auprc  += AUPRC

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fold_auc = auc(fpr, tpr)
            fold_roc_data.append((fpr, tpr, fold_auc))

            # 计算本折的PR曲线数据
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            fold_auprc = average_precision_score(y_test, y_prob)
            fold_pr_data.append((precision, recall, fold_auprc))
            fold_idx += 1
            
        ave_acc /= n_fold
        ave_prec /= n_fold
        ave_sens /= n_fold
        ave_f1_score /= n_fold
        ave_mcc /= n_fold
        ave_auc /= n_fold
        ave_auprc /= n_fold
        print('Final: ')
        print(f'  tp = \t{tp}')
        print(f'  fp = \t{fp}')
        print(f'  tn = \t{tn}')
        print(f'  fn = \t{fn}')
        print(f'  Acc = \t{ave_acc}')
        print(f'  prec = \t{ave_prec}')
        print(f'  sens = \t{ave_sens}')
        print(f'  f1_score = \t{ave_f1_score}')
        print(f'  MCC = \t{ave_mcc}')
        print(f'  AUC = \t{ave_auc}')
        print(f'  AUPRC = \t{ave_auprc}')
        f.write('Final: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(ave_acc)+'\t  prec = \t'+ str(ave_prec)+ '\t  sens = \t'+str(ave_sens)+'\t  f1_score = \t'+str(ave_f1_score)+ '\t  MCC = \t'+str(ave_mcc)+'\t  AUC = \t'+ str(ave_auc)+'\t  AUPRC = \t'+ str(ave_auprc)+'\n')

    overall_y_test = np.array(overall_y_test_list)
    overall_y_prob = np.array(overall_y_prob_list)

    # 绘制每一折曲线与最终结果曲线在一张图上
    plot_combined_curves(fold_roc_data, fold_pr_data)

if __name__ == "__main__":
    import random
    import torch
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    n_fold = 5
    CDA(n_fold)
    # plt.savefig('D:/auc_curve.png', dpi=300)
