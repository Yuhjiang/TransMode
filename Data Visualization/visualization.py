# -*- coding=utf-8 -*-
"""
Module:     VISUALIZATION
Summary:    分析数据，可视化
Author:     Yuhao Jiang
Created:    2018/01/29  Ver.1.0
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import smopy
from matplotlib.colors import LinearSegmentedColormap
import imageio
import datetime as dt
from geopy.distance import vincenty
import math
import csv


class Visualization(object):

    def __init__(self, segment_gps_features):
        """
        含有标注数据的GPS轨迹点
        :param segment_gps_label:
        """
        # segment_ID,trans_mode,former_trans_mode,latitude,longitude,date,time
        # Data\010_20080328145254_train,train,None,39.894505,116.321132,2008-03-28,14:55:14
        self.sg_gps = pd.read_csv(segment_gps_features)

    def heat_map(self, area, data_choosed=None, title=None):
        if type(area) != list:
            raise TypeError('area must be list!')

        smopy.TILE_SERVER = 'http://tile.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
        smopy.TILE_SIZE = 256
        map = smopy.Map((area[0], area[1], area[2], area[3]), z=9)
        data = self.sg_gps[self.sg_gps.trans_mode == data_choosed]