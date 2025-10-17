import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1️⃣ 读取 Excel 文件
file_path = r'C:\Users\鲁迅先生\Desktop\作业\pyth\@Python大数据分析与机器学习商业案例实战\@Python大数据分析与机器学习商业案例实战\第7章 K近邻算法\源代码汇总_Pycharm\葡萄酒.xlsx' # 这里换成你的文件路径
df = pd.read_excel(file_path)

# 打印前几行，查看数据格式
print("=== 原始数据预览 ===")
print(df.head())

# 2️⃣ 选择特征列和标签列
# 假设列名是：原始样本、酒精含量(%)、苹果酸含量(%)、分类
X = df[["酒精含量(%)", "苹果酸含量(%)"]]  # 特征
y = df["分类"]                              # 标签

# 3️⃣ 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=30
)

# 4️⃣ 数据标准化（Z-score 标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ 训练 KNN 模型
knn = KNeighborsClassifier(n_neighbors=2)  # K=3，可根据数据调整
knn.fit(X_train_scaled, y_train)

# 6️⃣ 预测并评估结果
y_pred = knn.predict(X_test_scaled)

print("\n=== 模型评估结果 ===")
print("准确率:", accuracy_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))
