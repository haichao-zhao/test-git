# -*- coding: utf-8 -*-

# @Time: 2019/11/9 3:29 PM

# @Project: test

# @Filename: feature_engineering.py

# @Author: zhaohaichao
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


# s1:satisfsction_level~~~False:MinMaxScaler;True:StandarScaler
# le:last_evaluation~~~False:MinMaxScaler;True:StandarScaler
# npr:number_project~~~False:MinMaxScaler;True:StandarScaler
# amh:average_monthly_hours~~~False:MinMaxScaler;True:StandarScaler
# tsc:time_spend_company~~~False:MinMaxScaler;True:StandarScaler
# wa:Work_accident~~~False:MinMaxScaler;True:StandarScaler
# pl5:promotion_last_5years~~~False:MinMaxScaler;True:StandarScaler
# dp:department~~~False:LabelEncoding;True:OneHotEncoding
# slr:salary~~~False:LabelEncoding;True:OneHotEncoding
def hr_preprocessing(s1=False, le=False, npr=False, amh=False, tsc=False, wa=False, pl5=False, dp=False, slr=False,
                     lower_d=False, ld_n=1):
    df = pd.read_csv("/Users/zhaohaichao/workspace/pythonspace/coding-185/data/HR.csv")

    # 清洗数据
    df = df.dropna(subset=['satisfaction_level', 'last_evaluation'])
    df = df[df['satisfaction_level'] <= 1][df['salary'] != 'nme']
    # 得到标注
    label = df['left']
    df = df.drop('left', axis=1)
    # TODU 特征选择

    # 特征处理
    scaler_lst = [s1, le, npr, amh, tsc, wa, pl5]
    column_lst = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours',
                  'time_spend_company', 'Work_accident', 'promotion_last_5years']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df[column_lst[i]] = StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[
                0]

    scaler_lst = [slr, dp]
    column_lst = ['salary', 'department']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i] == 'salary':
                df[column_lst[i]] = [map_salary(s) for s in df['salary'].values]
            else:
                df[column_lst[i]] = LabelEncoder().fit_transform(df[column_lst[i]])
        else:
            df = pd.get_dummies(df, columns=[column_lst[i]])
        df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]

    if lower_d:
        return PCA(n_components=ld_n).fit_transform(df.values), label
    return df, label


d = dict([('low', 0), ('medium', 1), ('high', 2)])


def map_salary(s):
    return d.get(s, 0)


def hr_modeling(features, label):
    from sklearn.model_selection import train_test_split
    f_v = features.values
    l_v = label.values

    X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)
    print(len(X_train), len(X_validation), len(X_test))

    # KNN
    from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    # 人工神经网络容器，层级化的
    from keras.models import Sequential
    # dense 稠密层 activation 激活函数
    from keras.layers.core import Dense, Activation
    # 随机梯度下降
    from keras.optimizers import SGD
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    #
    # --------------人工神经网络----------------
    mdl = Sequential()
    mdl.add(Dense(50, input_dim=len(f_v[0])))
    mdl.add(Activation('sigmoid'))
    mdl.add(Dense(2))
    mdl.add(Activation('softmax'))
    sgd = SGD(lr=0.01)
    mdl.compile(loss='mean_squared_error', optimizer='adam')
    mdl.fit(X_train, np.array([[0, 1] if i == 1 else [1, 0] for i in Y_train]), nb_epoch=10000, batch_size=8999)

    xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve, auc

    f = plt.figure()

    for i in range(len(xy_lst)):
        X_part = xy_lst[i][0]
        Y_part = xy_lst[i][1]
        Y_pred = mdl.predict(X_part)
        print(Y_pred)

        Y_pred = np.array(Y_pred[:, 1]).reshape((1, -1))[0]
        f.add_subplot(1, 3, i + 1)
        fpr, tpr, threshold = roc_curve(Y_part, Y_pred)
        plt.plot(fpr, tpr)
        print('NN', 'AUC', auc(fpr, tpr))
        print('NN', 'AUC_Score', roc_auc_score(Y_part, Y_pred))

    plt.show()
        # print('NN', '-ACC:', accuracy_score(Y_part, Y_pred))
        # print('NN', '-REC:', recall_score(Y_part, Y_pred))
        # print('NN', '-F1 :', f1_score(Y_part, Y_pred))

    return

    models = []
    # --------------分类----------------
    # # KNN
    # models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
    # # 朴素贝叶斯：必须是离散特征
    # # 高斯朴素贝叶斯：多用于符合高斯分布的特征
    # models.append(('GaussianNB', GaussianNB()))
    # # 伯努利朴素贝叶斯：多用于特征值为二值，就是0和1 的特征
    # models.append(('BernoulliNB', BernoulliNB()))
    # # 决策树
    # # 基尼系数
    # models.append(('DecisionTreeGini', DecisionTreeClassifier()))
    # # 熵增益
    # models.append(('DecisionTreeEntropy', DecisionTreeClassifier(criterion='entropy')))
    # # 装袋法：代表算法：随机森林
    # # 默认参数
    # models.append(('OriginalRandomForest', RandomForestClassifier()))
    # # 集成决策数数量为11 特征选择为全部特征
    # models.append(('RandomForest', RandomForestClassifier(n_estimators=11, max_features=None)))
    # # 提升法：代表算法：AdaBoost
    # models.append(('AdaBoost', AdaBoostClassifier()))

    # --------------回归----------------
    # 逻辑回归
    # models.append(('LogisticRegression', LogisticRegression(C=1000, tol=1e-10, solver='sag', max_iter=10000)))

    models.append(('GBDT', GradientBoostingClassifier(max_depth=6, n_estimators=100)))

    for clf_name, clf in models:
        clf.fit(X_train, Y_train)
        xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]

        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(i)
            print(clf_name, '-ACC:', accuracy_score(Y_part, Y_pred))
            print(clf_name, '-REC:', recall_score(Y_part, Y_pred))
            print(clf_name, '-F1 :', f1_score(Y_part, Y_pred))


# 线性回归
def regr_test(features, label):
    # print('X', features)
    # print('Y', label)
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    # regr = LinearRegression()
    # regr = Ridge(alpha=1)
    regr = Lasso(alpha=0.001)
    regr.fit(features.values, label.values)
    Y_pred = regr.predict(features.values)
    print('Coef:', regr.coef_)
    from sklearn.metrics import mean_squared_error
    print('MSE:', mean_squared_error(Y_pred, label.values))


def main():
    features, label = hr_preprocessing()
    # regr_test(features[['number_project', 'average_monthly_hours']], features['last_evaluation'])

    hr_modeling(features, label)


if __name__ == '__main__':
    main()
