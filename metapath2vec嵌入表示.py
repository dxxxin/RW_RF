import pickle
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np
from gensim.models import Word2Vec
from collections import deque


def generate_metapaths_by_rule(G, start_type, end_type, max_depth):
    metapaths = set()
    visited = set()

    for node in G.nodes:
        if G.nodes[node]['type'] == start_type:
            queue = deque()
            queue.append((node, [start_type]))
            visited.add(node)

            while queue:
                current_node, path = queue.popleft()
                if len(path) > max_depth:
                    continue
                if path[-1] == end_type:
                    metapaths.add(tuple(path))
                    continue
                for neighbor in G.neighbors(current_node):
                    if neighbor not in visited:
                        neighbor_type = G.nodes[neighbor]['type']
                        new_path = path + [neighbor_type]
                        queue.append((neighbor, new_path))
                        visited.add(neighbor)
    return list(metapaths)  # 将 metapaths 转换为 list



def hadamard_product_np(vec1, vec2):
    """Hadamard product optimized with NumPy."""
    return np.multiply(vec1, vec2)

def apply_metapath2vec_parallel(G, metapaths, dimensions=64, walk_length=50, num_walks=10, workers=64):
    """Optimized version with parallel random walks generation."""
    walks = []

    def generate_walks(node):
        walk_set = []
        for _ in range(num_walks):
            walk = [node]
            current_node = node
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current_node))
                if not neighbors:
                    break
                # 将 metapaths 转换为 list
                metapaths_list = list(metapaths)
                next_node_type = metapaths_list[(len(walk) % len(metapaths_list))]
                candidates = [n for n in neighbors if G.nodes[n]['type'] == next_node_type]
                if not candidates:
                    break  # 如果没有符合条件的节点，跳出当前游走
                next_node = random.choice(candidates)
                walk.append(next_node)
                current_node = next_node
            walk_set.append(walk)
        return walk_set

    # Using ThreadPoolExecutor to parallelize random walks generation
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_walks = {executor.submit(generate_walks, node): node for node in G.nodes()}
        for future in future_to_walks:
            walks.extend(future.result())

    # Train Word2Vec model on the generated walks
    model = Word2Vec(walks, vector_size=dimensions, window=10, min_count=1, sg=1, workers=workers)

    embeddings = {}
    for node in G.nodes():
        try:
            embeddings[node] = model.wv[node]
        except KeyError:
            print(f"Node {node} not found in model's vocabulary.")
    return embeddings


def apply_metapath2vec_concurrent(G, metapaths, num_workers=64):
    """Run metapath2vec in parallel."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future = executor.submit(apply_metapath2vec_parallel, G, metapaths)
        embeddings = future.result()
    return embeddings

def find_best_threshold_ROC(y_test, final_prediction):
    """Find the best threshold based on ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_test, final_prediction)
    J = tpr - fpr
    best_threshold_index = np.argmax(J)
    best_threshold = thresholds[best_threshold_index]
    return best_threshold

# Graph construction
G = nx.Graph()

df_genes = pd.read_excel("D:\\单个节点数据.xlsx", sheet_name='gene')
df_diss = pd.read_excel("D:\\单个节点数据.xlsx", sheet_name='dis')
df_herbs = pd.read_excel('D:\\单个节点数据.xlsx', sheet_name='herb')
df_ingredients = pd.read_excel('D:\\单个节点数据.xlsx', sheet_name='ingredient')

# Add nodes
for gene in df_genes['gene']:
    G.add_node(str(gene), type='gene')
for disease in df_diss['dis']:
    G.add_node(str(disease), type='dis')
for herb in df_herbs['herb']:
    G.add_node(str(herb), type='herb')
for ingredient in df_ingredients['ingredient']:
    G.add_node(str(ingredient), type='ingredient')

print(f"Number of nodes: {G.number_of_nodes()}")

# Read edge data and add edges
df_dis_gene_edges = pd.read_excel("D:\\节点数据12.1.xlsx", sheet_name='dis-gene')
df_dis_herb_edges = pd.read_excel('D:\\节点数据12.1.xlsx', sheet_name='dis-herb')
df_gene_gene_edges = pd.read_excel('D:\\节点数据12.1.xlsx', sheet_name='gene-gene')
df_herb_gene_edges = pd.read_excel('D:\\节点数据12.1.xlsx', sheet_name='herb-gene')
df_ingredient_gene_edges = pd.read_excel('D:\\节点数据12.1.xlsx', sheet_name='ingredient-gene')
df_ingredient_herb_edges = pd.read_excel('D:\\节点数据12.1.xlsx', sheet_name='herb-ingredient')
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

# Generate metapaths
# start_type = "ingredient"
# end_type = "dis"
# max_depth = 4
# metapaths = generate_metapaths_by_rule(G, start_type, end_type, max_depth)
# print("自动生成的元路径：")
# for path in metapaths:
#     print(path)
# print(f'metapaths: {len(metapaths)}')
metapaths = [['ingredient', 'gene', 'dis'],
             ['ingredient', 'herb', 'dis'],
             ['dis', 'gene', 'ingredient'],
             ['dis', 'herb', 'ingredient'],
             ['ingredient', 'herb', 'gene', 'dis']]

# Get embeddings
embeddings = apply_metapath2vec_concurrent(G, metapaths)

# Further processing with embeddings can be done here

print(f"Number of embeddings: {len(embeddings)}")

# 读取二值化矩阵
file_path = "D:\\二值化矩阵_行列名.xlsx"
binary_matrix = pd.read_excel(file_path, index_col=0)

# 将行索引和列转换为字符串
binary_matrix.index = binary_matrix.index.map(str)
binary_matrix.columns = binary_matrix.columns.map(str)


# 确认索引和列中数据与嵌入向量的匹配
ingredient_vectors = {str(node): embeddings[str(node)] for node in binary_matrix.index if str(node) in embeddings}
dis_vectors = {str(node): embeddings[str(node)] for node in binary_matrix.columns if str(node) in embeddings}

# 将嵌入向量保存为 Excel 文件
ingredient_df = pd.DataFrame.from_dict(ingredient_vectors, orient='index')
dis_df = pd.DataFrame.from_dict(dis_vectors, orient='index')

ingredient_df.to_excel("D:\\metapath2vec\\ingredient_vectors64.xlsx")
dis_df.to_excel("D:\\metapath2vec\\dis_vectors64.xlsx")

