import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1️⃣ 读取 Excel 数据
file_path = r'C:\Users\鲁迅先生\Desktop\作业\pyth\@Python大数据分析与机器学习商业案例实战\@Python大数据分析与机器学习商业案例实战\第7章 K近邻算法\源代码汇总_Pycharm\手写字体识别.xlsx'
df = pd.read_excel(file_path)

# 2️⃣ 查看前几行确认数据
print("数据预览：")
print(df.head())

# 3️⃣ 特征和标签
X = df.iloc[:, 1:].values  # 第2列到第1024列，像素特征
y = df.iloc[:, 0].values   # 第一列，数字标签

# 4️⃣ 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5️⃣ 特征标准化（KNN 对距离敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ 训练 KNN 模型
knn = KNeighborsClassifier(n_neighbors=3)  # K=3，可调整
knn.fit(X_train_scaled, y_train)

# 7️⃣ 模型预测
y_pred = knn.predict(X_test_scaled)

# 8️⃣ 输出结果
print("\n=== 模型评估 ===")
print("准确率:", accuracy_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred, zero_division=0))
