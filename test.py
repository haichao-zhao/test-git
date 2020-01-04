# -*- coding: utf-8 -*-

# @Time: 2019/11/8 10:24 PM

# @Project: test

# @Filename: test.py

# @Author: zhaohaichao

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/zhaohaichao/workspace/pythonspace/coding-185/data/HR.csv")
df = df.dropna()
df = df[df["department"] != "sale"][df["salary"] != "nme"]

sns.set(style="darkgrid")
sns.set_palette(sns.color_palette("RdBu", n_colors=7))

# ---------------------分割线-----------------------------------------------------
# 饼图

lbs = df['department'].value_counts().index
explodes = [0.08 if i == 'sales' else 0 for i in lbs]
plt.pie(df['department'].value_counts(normalize=True), labels=lbs, autopct='%1.1f%%', colors=sns.color_palette('Reds'),
        explode=explodes)

# ---------------------分割线-----------------------------------------------------
# 折线图
# sub_df = df.groupby("time_spend_company").mean()
# sns.pointplot(x=sub_df.index, y=sub_df["left"])

# sns.pointplot(x="time_spend_company", y="left", data=df)

# ---------------------分割线-----------------------------------------------------
# 箱线图
# sns.boxplot(y=df["time_spend_company"])
# sns.boxplot(x=df["time_spend_company"], saturation=0.75, whis=3)

# ---------------------分割线-----------------------------------------------------
# 直方图
# f = plt.figure()
# f.add_subplot(1, 3, 1)
# sns.distplot(df["satisfaction_level"], bins=10)
# f.add_subplot(1, 3, 2)
# sns.distplot(df["last_evaluation"], bins=10)
# f.add_subplot(1, 3, 3)
# sns.distplot(df["average_monthly_hours"], bins=10)

# ---------------------分割线-----------------------------------------------------
# 柱状图
# sns.countplot(x="salary", data=df, hue="department")

# plt.title("SALARY")
# plt.xlabel("salary")
# plt.ylabel("count")
# plt.xticks(np.arange(len(df["salary"].value_counts())) + 0.5, df["salary"].value_counts().index)
# plt.axis([0, 3, 0, 10000])
#
# plt.bar(np.arange(len(df["salary"].value_counts())) + 0.5, df["salary"].value_counts(), width=0.5)
# for x, y in zip(np.arange(len(df["salary"].value_counts())) + 0.5, df["salary"].value_counts()):
#     plt.text(x, y, y, ha="center", va="bottom")
# ---------------------分割线-----------------------------------------------------
plt.show()
