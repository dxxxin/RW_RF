import numpy as np
import pandas as pd
import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve


def find_best_threshold_ROC(y_test,final_prediction):
    fpr,tpr,thresholds = roc_curve(y_test,final_prediction)
    J = tpr - fpr
    best_threshold_index = np.argmax(J)
    best_threshold = thresholds[best_threshold_index]
    return best_threshold

def hadamard_product(vec1, vec2):
    return np.multiply(vec1, vec2)

file_path = "二值化矩阵_行列名.xlsx"
binary_matrix = pd.read_excel(file_path, index_col=0)



# 读取 ingredient_vectors 的 Excel 文件
ingredient_df = pd.read_excel('ingredient.xlsx')
ingredient_vectors = {}
# 遍历每一行
for index, row in ingredient_df.iterrows():
    node = str(row['node'])  # 提取 node 列的值
    # 提取除了 'node' 列之外的所有列作为向量
    vector = [row[col] for col in ingredient_df.columns if col != 'node']
    ingredient_vectors[node] = vector  # 将 node 和向量存储到字典中


# 读取 dis_vectors 的 Excel 文件
dis_df = pd.read_excel('dis.xlsx')
dis_vectors = {}
for index, row in dis_df.iterrows():
    node = str(row['node'])
    vector = [row[col] for col in dis_df.columns if col != 'node']
    dis_vectors[node] = vector  # 将 node 和向量存储到字典中


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

# 分离正负样本
positive_samples = [sample for sample in samples_info if sample[3] == 1]
negative_samples = [sample for sample in samples_info if sample[3] == 0]


# 合并正负样本
balanced_samples = positive_samples + negative_samples
random.shuffle(balanced_samples)


with open("成分疾病.pkl", "wb") as tf:
    pickle.dump(balanced_samples,tf)



with open("成分疾病.pkl", "rb") as tf:
    balanced_samples = pickle.load(tf)

# 提取特征和标签
X = np.array([sample[2] for sample in balanced_samples])
y = np.array([sample[3] for sample in balanced_samples])

# 划分训练集和测试集，并保持样本信息和标签的一致性
X_train, X_test, y_train, y_test, samples_info_train, samples_info_test = train_test_split(
    X, y, balanced_samples, test_size=0.2, stratify=y, random_state=42
)

# 将 y_test 添加回 samples_info_test
samples_info_test = [(info[0], info[1], info[2], label) for info, label in zip(samples_info_test, y_test)]

# 训练和评估模型
rf = RandomForestClassifier(n_estimators=20, random_state=42)
rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# y_proba = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict_proba(X_test)[:, 1]



best_threshold = find_best_threshold_ROC(y_test, y_pred)
print("best_threshold:",best_threshold)
y_pred =(y_pred > best_threshold).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
# auc_score = roc_auc_score(y_test, y_proba)
# aupr_score = average_precision_score(y_test, y_proba)

auc_score = roc_auc_score(y_test, y_pred)
aupr_score = average_precision_score(y_test, y_pred)

print(f"AUC: {auc_score:.4f}")
print(f"AUPR: {aupr_score:.4f}")


# 输出结果
results_df = pd.DataFrame({
    'Ingredient': [info[0] for info in samples_info_test],
    'Disease': [info[1] for info in samples_info_test],
    'True_Label': [info[3] for info in samples_info_test],
    'Predicted_Prob': y_pred
})
results_df.to_csv("predictions_probabilities.csv", index=False)


print(f"X_train: {len(X_train)}, X_test: {len(X_test)}")
print(f"y_train: {len(y_train)}, y_test: {len(y_test)}")
# print(y_train)
print(f"samples_info_train: {len(samples_info_train)}, samples_info_test: {len(samples_info_test)}")

