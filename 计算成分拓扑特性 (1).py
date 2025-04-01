
import networkx as nx
import pandas as pd
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# 读取图数据
with open('图数据文件.pkl', 'rb') as f:
    G = pickle.load(f)
print("图数据加载完成。")

# 预先计算所有中心性度量
print("开始计算中心性度量...")
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=5000, tol=1e-06)
pagerank = nx.pagerank(G)
clustering_coefficient = nx.clustering(G)
k_shell_decomposition = nx.core_number(G)
harmonic_centrality = nx.harmonic_centrality(G)
load_centrality = nx.load_centrality(G)

# 社区检测
communities = list(nx.community.louvain_communities(G, weight='weight'))

def get_community_index(node, communities):
    for idx, community in enumerate(communities):
        if node in community:
            return idx
    return -1

def calculate_feature_for_node(node, G, degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality, pagerank, clustering_coefficient, k_shell_decomposition, harmonic_centrality, load_centrality, communities):
    data = {
        'node': node,
        'degree': G.degree(node),
        'degree_centrality': degree_centrality[node],
        'closeness_centrality': closeness_centrality[node],
        'pagerank': pagerank[node],
        'clustering_coefficient': clustering_coefficient[node]
    }
    return data

# 获取所有成分节点
ingredient_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'ingredient']

# 获取所有疾病节点
disease_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'dis']

# 使用线程池并行计算特征
def parallel_feature_calculation(nodes, G, degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality, pagerank, clustering_coefficient, k_shell_decomposition, harmonic_centrality, load_centrality, communities, max_workers=20):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_node = {executor.submit(calculate_feature_for_node, node, G, degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality, pagerank, clustering_coefficient, k_shell_decomposition, harmonic_centrality, load_centrality, communities): node for node in nodes}
        results = []
        for future in as_completed(future_to_node):
            node = future_to_node[future]
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                print(f'生成器 {node} 产生了异常: {exc}')
    return pd.DataFrame(results)

print("开始计算成分节点的特征...")
df_ingredient_features = parallel_feature_calculation(ingredient_nodes, G, degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality, pagerank, clustering_coefficient, k_shell_decomposition, harmonic_centrality, load_centrality, communities)

print("开始计算疾病节点的特征...")
df_disease_features = parallel_feature_calculation(disease_nodes, G, degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality, pagerank, clustering_coefficient, k_shell_decomposition, harmonic_centrality, load_centrality, communities)

# 重命名列，区分节点类型
df_ingredient_features = df_ingredient_features.rename(columns={'node': 'ingredient'})
df_disease_features = df_disease_features.rename(columns={'node': 'dis'})

# 保存特征到Excel文件
print("保存特征到Excel文件...")
with pd.ExcelWriter('D:\\贾帅兵\\成分疾病网络拓扑节点特征.xlsx') as writer:
    df_ingredient_features.to_excel(writer, sheet_name='ingredient_features', index=False)
    df_disease_features.to_excel(writer, sheet_name='dis_features', index=False)

print("特征计算完成并保存到文件。")