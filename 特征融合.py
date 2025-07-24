import pandas as pd

dis_file = 'dis_vectors_deepwalk.xlsx'
ing_file = 'ingredient_vectors_deepwalk.xlsx'

# 读取文件1和文件2
file1_path = '成分疾病网络拓扑节点特征0401.xlsx'
file2_path = ing_file
file3_path = dis_file

# 读取文件1
df1_ingredient = pd.read_excel(file1_path, sheet_name="ingredient_features")
df1_dis = pd.read_excel(file1_path, sheet_name="dis_features")

# 读取文件2，并获取 ingredient 列
df2_ingredient = pd.read_excel(file2_path)
#df2_ingredient = df2_ingredient.rename(columns={0: 'ingredient'})
df3_dis = pd.read_excel(file3_path)
#df3_dis = df3_dis.rename(columns={0: 'dis'})

ingredient_order = df2_ingredient['ingredient'].tolist()
dis_order = df3_dis['dis'].tolist()

df1_ingredient_sorted = df1_ingredient.set_index('ingredient').loc[ingredient_order].reset_index()
df1_dis_sorted = df1_dis.set_index('dis').loc[dis_order].reset_index()

# 保存排序后的文件
output_ingredient_path = 'ingredient_features_sorted.xlsx'
df1_ingredient_sorted.to_excel(output_ingredient_path, index=False)

output_dis_path = 'dis_features_sorted.xlsx'
df1_dis_sorted.to_excel(output_dis_path, index=False)

print("文件1已按照文件2的 dis 列排序并保存到:", output_ingredient_path)
print("文件1已按照文件2的 dis 列排序并保存到:", output_dis_path)



#成分特征融合
# 文件路径
file_a_path_ingredient = ing_file
file_b_path_ingredient = 'ingredient_features_sorted.xlsx'
output_path_ingredient = 'ingredient.xlsx'

# 读取文件（自动识别首行为列名，首列为索引）
df_a_ingredient = pd.read_excel(file_a_path_ingredient, index_col=0)
df_b_ingredient = pd.read_excel(file_b_path_ingredient, index_col=0)

# 自动生成列名逻辑
# 获取文件A的列数，并生成递增列名（例如：文件A有3列，则文件B列名自动变为4）
#auto_new_col = df_a_ingredient.shape[1] + 1  # 自动计算新列起始值
# 如果文件B有多列，可批量更新列名（示例处理单列情况）
#df_b_ingredient.columns = [f"{auto_new_col}"]  # 单列时保持列名为字符串
#df_b_ingredient.columns = [0]
# 横向合并（自动对齐索引）
merged_df = pd.concat([df_a_ingredient, df_b_ingredient], axis=1)

# 保存结果（包含原始列名和索引）
merged_df.to_excel(output_path_ingredient)


#疾病特征融合
# 文件路径
file_a_path_dis = dis_file
file_b_path_dis = 'dis_features_sorted.xlsx'
output_path_dis = 'dis.xlsx'

# 读取文件（自动识别首行为列名，首列为索引）
df_a_dis = pd.read_excel(file_a_path_dis, index_col=0)
df_b_dis = pd.read_excel(file_b_path_dis, index_col=0)

# 自动生成列名逻辑
# 获取文件A的列数，并生成递增列名（例如：文件A有3列，则文件B列名自动变为4）
#auto_new_col = df_a_dis.shape[1] + 1  # 自动计算新列起始值

# 如果文件B有多列，可批量更新列名（示例处理单列情况）
#df_b_dis.columns = [f"{auto_new_col}"]  # 单列时保持列名为字符串

# 横向合并（自动对齐索引）
merged_df = pd.concat([df_a_dis, df_b_dis], axis=1)

# 保存结果（包含原始列名和索引）
merged_df.to_excel(output_path_dis)