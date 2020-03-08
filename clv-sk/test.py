from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import tree

datas = datasets.load_boston()
data_X = datas.data
data_y = datas.target
#1：线性回归预测房价
model = LinearRegression() 
model.fit(data_X,data_y) #训练
print(model.predict(data_X[:4,:])) #预测
print(data_y[:4])



iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.3) #训练集测试集划分：留出法、k交叉法、自助法
# 2：K近邻分类
knn = KNeighborsClassifier()
# 3：决策树
#dtc = tree.DecisionTreeClassifier()
knn.fit(X_train,y_train) #训练

print(knn.predict(X_test))#使用模型预测y值
print(y_test) # 打印真实标签值y',与y作比对
