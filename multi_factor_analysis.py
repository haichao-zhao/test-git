# -*- coding: utf-8 -*-

# @Time: 2019/11/9 1:03 PM

# @Project: test

# @Filename: multi_factor_analysis.py

# @Author: zhaohaichao

from statsmodels.graphics.api import qqplot
from matplotlib import pyplot as plt
import scipy.stats as ss

k = ss.norm.rvs(size=100)
j = qqplot(data=k)

plt.show()

