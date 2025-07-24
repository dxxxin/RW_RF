import numpy as np
import pandas as pd
import random
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve
from node2vec import Node2Vec
from gensim.models import Word2Vec

def find_best_threshold_ROC(y_test, final_prediction):
    fpr, tpr, thresholds = roc_curve(y_test, final_prediction)
    J = tpr - fpr
    best_threshold_index = np.argmax(J)
    best_threshold = thresholds[best_threshold_index]
    return best_threshold

def hadamard_product(vec1, vec2):
    return np.multiply(vec1, vec2)

def apply_node2vec_concurrent(G, p=1, q=1):
    node2vec = Node2Vec(G, dimensions=64, walk_length=50, num_walks=10, workers=1, p=p, q=q)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = {}
    for node in G.nodes():
        try:
            embeddings[node] = model.wv[node]
        except KeyError:
            print(f"Node {node} not found in model's vocabulary.")
    return embeddings

def apply_node2vec(G, p=1, q=1, num_workers=2):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future = executor.submit(apply_node2vec_concurrent, G, p, q)
        embeddings = future.result()
    return embeddings


def random_walk(graph, start_node, walk_length):
    walk = [str(start_node)]
    while len(walk) < walk_length:
        current_node = walk[-1]
        neighbors = list(graph.neighbors(current_node))
        if len(neighbors) == 0:
            break
        next_node = random.choice(neighbors)
        walk.append(str(next_node))
    return walk

# Step 3: 执行多次随机游走
def generate_walks(graph, num_walks, walk_length):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(graph, node, walk_length))
    return walks

# Step 4: 训练 Word2Vec 模型
def train_deepwalk(graph, walk_length=10, num_walks=50, dimensions=64, window_size=2, workers=4, epochs=5):
    walks = generate_walks(graph, num_walks, walk_length)
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,         # 使用 skip-gram
        workers=workers,
        epochs=epochs
    )
    return model

# Step 5: 获取嵌入向量
def get_node_embeddings(model):
    embeddings = {}
    for node in model.wv.index_to_key:
        embeddings[node] = model.wv[node]
    return embeddings

def apply_deepwalk(G):
    model = train_deepwalk(G)
    embeddings = get_node_embeddings(model)
    return embeddings

def main():
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

    # 读取嵌入
    #embeddings = apply_node2vec(G, p=0.5, q=4)
    embeddings = apply_deepwalk(G)

    #print(f"Number of embeddings: {len(embeddings)}")
    #print(embeddings[0])
    file_path = "二值化矩阵_行列名.xlsx"
    binary_matrix = pd.read_excel(file_path, index_col=0)

    # 将行索引和列转换为字符串
    binary_matrix.index = binary_matrix.index.map(str)
    binary_matrix.columns = binary_matrix.columns.map(str)

    # 打印 binary_matrix 中的索引和列名称
    #print("Binary Matrix Index (Ingredients):", binary_matrix.index.tolist())
    #print("Binary Matrix Columns (Diseases):", binary_matrix.columns.tolist())

    # 确认索引和列中数据与嵌入向量的匹配
    ingredient_vectors = {str(node): embeddings[str(node)] for node in binary_matrix.index if str(node) in embeddings}
    dis_vectors = {str(node): embeddings[str(node)] for node in binary_matrix.columns if str(node) in embeddings}

    print(ingredient_vectors)
    # 将嵌入向量保存为 Excel 文件
    ingredient_df = pd.DataFrame.from_dict(ingredient_vectors, orient='index')
    #ingredient_df = ingredient_df.rename(columns={0: 'ingredient'})
    dis_df = pd.DataFrame.from_dict(dis_vectors, orient='index')
    #dis_df = dis_df.rename(columns={0: 'dis'})

    ingredient_df.to_excel("ingredient_vectors_deepwalk.xlsx")
    dis_df.to_excel("dis_vectors_deepwalk.xlsx")

if __name__ == "__main__":
    main()
