import os
import numpy as np
import pandas as pd
import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def find_best_threshold_ROC(y_test,final_prediction):
    fpr,tpr,thresholds = roc_curve(y_test,final_prediction)
    J = tpr - fpr
    best_threshold_index = np.argmax(J)
    best_threshold = thresholds[best_threshold_index]
    return best_threshold

def hadamard_product(vec1, vec2):
    return np.multiply(vec1, vec2)

def get_ing_dis_info(premodel):
    if "all_sample_"+premodel+".pkl" in os.listdir('data/'):
        print('existing samples info..............')
        with open("data/all_sample_"+premodel+".pkl", "rb") as tf:
            samples_info = pickle.load(tf)
        return samples_info

    print('reading...............')
    file_path = "二值化矩阵_行列名.xlsx"
    binary_matrix = pd.read_excel(file_path, index_col=0)

    # 读取 ingredient_vectors 的 Excel 文件
    ingredient_df = pd.read_excel('ingredient.xlsx')
    ingredient_vectors = {}
    # 遍历每一行
    for index, row in ingredient_df.iterrows():
        #print(row)
        #print(row[0])
        node = str(row['ingredient'])  # 提取 node 列的值
        # 提取除了 'node' 列之外的所有列作为向量
        vector = [row[col] for col in ingredient_df.columns if col != 'ingredient']
        ingredient_vectors[node] = vector  # 将 node 和向量存储到字典中
    print('finish ingredient_df')

    # 读取 dis_vectors 的 Excel 文件
    dis_df = pd.read_excel('dis.xlsx')
    dis_vectors = {}
    for index, row in dis_df.iterrows():
        #print(row)
        node = str(row['dis'])
        vector = [row[col] for col in dis_df.columns if col != 'dis']
        dis_vectors[node] = vector  # 将 node 和向量存储到字典中
    print('finish dis_df')

    print(f"Number of ingredient vectors: {len(ingredient_vectors)}")
    print(f"Number of disease vectors: {len(dis_vectors)}")

    samples_info = []
    # 生成样本及其信息
    for ingredient in binary_matrix.index:
        for dis in binary_matrix.columns:
            if str(ingredient) in ingredient_vectors and str(dis) in dis_vectors:
                hadamard_vec = hadamard_product(ingredient_vectors[str(ingredient)], dis_vectors[str(dis)])
                label = binary_matrix.loc[ingredient, dis]
                samples_info.append((ingredient, dis, hadamard_vec, label))
    print(f"Total samples: {len(samples_info)}")

    with open("data/all_sample_"+premodel+".pkl", "wb") as tf:
        pickle.dump(samples_info,tf)

    return samples_info

def run_predict(samples_info,premodel):
    kf = KFold(n_splits=5, shuffle=True,random_state=42)

    # 分离正负样本
    positive_samples = [sample for sample in samples_info if sample[3] == 1]
    negative_samples = [sample for sample in samples_info if sample[3] == 0]

    fold = 1
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(positive_samples),
                                                                            kf.split(negative_samples)):
        print('----------------fold ',fold,'----------------')
        fold+=1

        train_index = np.concatenate((train_pos_idx, train_neg_idx), axis=0)
        random.shuffle(train_index)
        X_train = np.array([samples_info[i][2] for i in train_index])
        y_train = np.array([samples_info[i][3] for i in train_index])

        test_index = np.concatenate((test_pos_idx, test_neg_idx), axis=0)
        random.shuffle(test_index)
        X_test = np.array([samples_info[i][2] for i in test_index])
        y_test = np.array([samples_info[i][3] for i in test_index])

        #random forest
        rf = RandomForestClassifier(n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_test)[:, 1]
        #print(y_pred)
        #svm
        #svm = SVC(kernel="linear")
        #svm.fit(X_train, y_train)
        #y_pred = svm.predict_proba(X_test)[:, 1]
        # Logistic
        #logistic = LogisticRegression(solver='lbfgs', max_iter=500)
        #logistic.fit(X_train, y_train)
        #y_pred = logistic.predict_proba(X_test)[:, 1]
        #xgboost
        #xgboost = XGBClassifier(max_depth=3,learning_rate=0.1,n_estimators=100)
        #xgboost.fit(X_train, y_train)
        #y_pred = xgboost.predict_proba(X_test)[:, 1]

        best_threshold = find_best_threshold_ROC(y_test, y_pred)
        print("best_threshold:", best_threshold)
        y_pred = (y_pred > best_threshold).astype(int)

        print("Classification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        auc_score = roc_auc_score(y_test, y_pred)
        aupr_score = average_precision_score(y_test, y_pred)
        print(f"AUC: {auc_score:.4f}")
        print(f"AUPR: {aupr_score:.4f}")


def get_result():
    with open("data/y_test.pkl", "rb") as tf:
        y_test = pickle.load(tf)
    with open("data/y_predict.pkl", "rb") as tf:
        y_pred = pickle.load(tf)

    print("Classification Report:")

    best_threshold = find_best_threshold_ROC(y_test, y_pred)
    print("best_threshold:", best_threshold)
    y_pred = (y_pred > best_threshold).astype(int)

    print(classification_report(y_test, y_pred, digits=4))
    auc_score = roc_auc_score(y_test, y_pred)
    aupr_score = average_precision_score(y_test, y_pred)
    print(f"AUC: {auc_score:.4f}")
    print(f"AUPR: {aupr_score:.4f}")

premodel = 'deepwalk'
samples_info = get_ing_dis_info(premodel)
run_predict(samples_info,premodel)
