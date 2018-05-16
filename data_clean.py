# -*- coding=utf-8 -*-
"""
Module:     data_clearn
Summary:    对数据重新整合，清洗
Author:     Yuhao Jiang
Created:    2018/05/16  Ver.1.0
"""
import xlrd
import xlwt
import pandas as pd
import numpy as np
from utils import *

"""
处理trip数据：
1. 剔除测试数据，不合理的GPS数据，修正一些可修复数据
2. 为每一个subtrip添加距离
3. 按日期重新分配
"""

path = r'D:\Zhejiang University\Graduate Project\Data\Data\trip.csv'
data = pd.read_csv(path).iloc[:, 1:]
"""
数据sample:
用户ID,日期,次数,出行目的,出发地点,达到地点,出发时间,达到时间,出行方式1,用时1,出行方式2,用时2,出行方式3,用时3,出行方式4,用时4,出行路线,其他
460FAD4B-1357-45A6-8AEB-504F8E716CAD,4/17/2018,1,2,食堂,建工实验大厅,8:10:00,8:15:00,2,5, , , , , , ,6-19-30, 
"""
columns = ['用户ID', '星期', '日期', 'Trip_ID', '出行目的', '第几次出行', '出发地点', '到达地点',
           '出发时间', '达到时间', '出发时', '出发分', '到达时', '到达分', '出行时间',
           '出行方式总数', '主要出行方式', '出行方式1', '用时1', '出行方式2', '用时2', '出行方式3', '用时3', '出行方式4', '用时4',
           '性别', '年龄', '宿舍区', '年级', '专业大类', '自行车保有', '电动车保有', '汽车保有',
           '校内主要出行方式', '校外主要出行方式']
