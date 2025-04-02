from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import networkx as nx
import pickle

G = nx.Graph()
df_genes = pd.read_excel("单个节点数据.xlsx", sheet_name='gene')
df_diss = pd.read_excel("单个节点数据.xlsx", sheet_name='dis')
df_herbs = pd.read_excel('单个节点数据.xlsx', sheet_name='herb')
df_ingredients = pd.read_excel('单个节点数据.xlsx', sheet_name='ingredient')
# 将所有节点标识符统一为字符串类型
for gene in df_genes['gene']:
    G.add_node(str(gene), type='gene')
for disease in df_diss['dis']:
    G.add_node(str(disease), type='dis')
for herb in df_herbs['herb']:
    G.add_node(str(herb), type='herb')
for ingredient in df_ingredients['ingredient']:
    G.add_node(str(ingredient), type='ingredient')
print(f"Number of nodes: {G.number_of_nodes()}")
df_dis_gene_edges = pd.read_excel("节点数据12.1.xlsx", sheet_name='dis-gene')
df_dis_herb_edges = pd.read_excel('节点数据12.1.xlsx', sheet_name='dis-herb')
df_gene_gene_edges = pd.read_excel('节点数据12.1.xlsx', sheet_name='gene-gene')
df_herb_gene_edges = pd.read_excel('节点数据12.1.xlsx', sheet_name='herb-gene')
df_ingredient_gene_edges = pd.read_excel('节点数据12.1.xlsx', sheet_name='ingredient-gene')
df_ingredient_herb_edges = pd.read_excel('节点数据12.1.xlsx', sheet_name='herb-ingredient')
for index, row in df_dis_gene_edges.iterrows():
    G.add_edge(str(row['dis']), str(row['gene']))
for index, row in df_dis_herb_edges.iterrows():
    G.add_edge(str(row['herb']), str(row['dis']))
for index, row in df_gene_gene_edges.iterrows():
    G.add_edge(str(row['gene1']), str(row['gene2']))
for index, row in df_herb_gene_edges.iterrows():
    G.add_edge(str(row['herb']), str(row['gene']))
for index, row in df_ingredient_gene_edges.iterrows():
    G.add_edge(str(row['ingredient']), str(row['gene']))
for index, row in df_ingredient_herb_edges.iterrows():
    G.add_edge(str(row['herb']), str(row['ingredient']))
print(f"Number of edges: {G.number_of_edges()}")

# 预先计算所有中心性度量
print("开始计算中心性度量...")
closeness_centrality = nx.closeness_centrality(G)
pagerank = nx.pagerank(G)
clustering_coefficient = nx.clustering(G)

def calculate_feature_for_node(node, G, closeness_centrality, pagerank, clustering_coefficient):
    data = {
        'node': node,
        'degree': G.degree(node),
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
def parallel_feature_calculation(nodes, G, closeness_centrality, pagerank, clustering_coefficient, max_workers=20):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_node = {executor.submit(calculate_feature_for_node, node, G, closeness_centrality, pagerank, clustering_coefficient): node for node in nodes}
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
df_ingredient_features = parallel_feature_calculation(ingredient_nodes, G, closeness_centrality, pagerank, clustering_coefficient)

print("开始计算疾病节点的特征...")
df_disease_features = parallel_feature_calculation(disease_nodes, G, closeness_centrality, pagerank, clustering_coefficient)

# 重命名列，区分节点类型
df_ingredient_features = df_ingredient_features.rename(columns={'node': 'ingredient'})
df_disease_features = df_disease_features.rename(columns={'node': 'dis'})

# 保存特征到Excel文件
print("保存特征到Excel文件...")
with pd.ExcelWriter('成分疾病网络拓扑节点特征0401.xlsx') as writer:
    df_ingredient_features.to_excel(writer, sheet_name='ingredient_features', index=False)
    df_disease_features.to_excel(writer, sheet_name='dis_features', index=False)

print("特征计算完成并保存到文件。")