import sys
sys.getfilesystemencoding()
sys._enablelegacywindowsfsencoding()
sys.getfilesystemencoding()

# 导入需要的库
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
import matplotlib.pylab as plt


# 导入数据
train = pd.read_csv("F:\work\百果科技\1 AI\4 智能选址\1自研\4 建模分析结果\Python\选址模型第一版//test_python.csv",encoding='gbk')
#print(train)
target='实际销售额'                        # 实际销售额为预测目标
IDcol= ['门店编号', '店名']               # 待过滤字段


# 准备特征集和目标列表
x_columns = [x for x in train.columns if x not in [target]+IDcol]
X = train[x_columns]
y = train[target]

city_mapping = dict(zip(np.unique(X['城市']),range(len(np.unique(X['城市'])))))
X['城市']= X['城市'].map(city_mapping)

orgps_mapping = dict(zip(np.unique(X['配送中心']),range(len(np.unique(X['配送中心'])))))
X['配送中心']= X['配送中心'].map(orgps_mapping)

shoppingd_mapping = dict(zip(np.unique(X['商圈结构']),range(len(np.unique(X['商圈结构'])))))
X['商圈结构']= X['商圈结构'].map(shoppingd_mapping)

# scikit-learn已经有了模型持久化 （将训练好的模型保存下来，以方便后期直接调用）
from sklearn.externals import joblib
import os
os.chdir("E:/百果园工作/选址建模测试/第一阶段/randomForest_python")
forest = RandomForestRegressor(n_estimators = 1000,criterion='mse',random_state=1,n_jobs=-1)
forest.fit(X,y)
joblib.dump(forest, "train_model.m")

# 模型调回
from sklearn.externals import joblib
import os
os.chdir("E:/百果园工作/选址建模测试/第一阶段/randomForest_python")
forest = joblib.load("train_model.m")


