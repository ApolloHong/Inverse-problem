# 设置随机种子
seed = 123

# 导入相应的包和模块
import pandas as pd
import numpy as np
from seaborn import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# =加载数据集，我们随机选取了5组数据进行测试
df = load_dataset('tips').drop(columns=['tip', 'sex']).sample(n=5, random_state=seed)

# 增加一些缺失值用于测试
df.iloc[[1, 2, 4], [2, 4]] = np.nan
print(df)

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['total_bill', 'size']),
                                                    df['total_bill'],
                                                    test_size=.2,
                                                    random_state=seed)

# 构建pipeline
pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
pipe.fit(X_train)

# 处理和处理之前训练数据对比
print("******************** Training data ********************")
print(X_train)
print(pd.DataFrame(pipe.transform(X_train), columns=pipe['encoder'].get_feature_names(X_train.columns)))

# 处理和处理之前的测试集对比
print("******************** Test data ********************")
print(X_test)
print(pd.DataFrame(pipe.transform(X_test), columns=pipe['encoder'].get_feature_names(X_train.columns)))