# -*- coding=utf-8 -*-
"""
Module:     VISUALIZATION
Summary:    分析数据，可视化
Author:     Yuhao Jiang
Created:    2018/01/29  Ver.1.0
Update:     2018/03/13  Ver.1.1
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import smopy
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import imageio
import datetime as dt
from geopy.distance import vincenty
import math
from math import pi
import csv
from scipy import ndimage
import gmap
import imageio

ee = 0.00669342162296594323     # 偏心率平方
a = 6378245.0                   # 长半轴


class Visualization(object):

    def __init__(self, gps_data):
        """
        含有标注数据的GPS轨迹点
        :param segment_gps_label:
        """
        # segment_ID,trans_mode,former_trans_mode,latitude,longitude,date,time
        # Data\010_20080328145254_train,train,None,39.894505,116.321132,2008-03-28,14:55:14
        self.data = pd.read_csv(gps_data)

    @staticmethod
    def heatmap(gps, area, bins=200, zoom=9, smoothing=1, vmax=4, title=None, show=False):
        """
        生成热力图
        :param lat: gps的经度
        :param long: gps的纬度
        :param area: 显示范围的GPS点
        :param bins:    尺寸
        :param smoothing: 设置平滑
        :param vmax:
        :param title:   图名
        :return:
        """
        # 生成范围地图
        mapZoom = smopy.Map((area[0], area[1], area[2], area[3]), z=zoom)

        # 提取范围内的GPS点
        gps_data = gps[gps.latitude.between(area[0], area[2]) & gps.longitude.between(area[1], area[3])]
        lat = gps_data.latitude
        long = gps_data.longitude
        x, y = mapZoom.to_pixels(lat, long)

        ax = mapZoom.show_mpl(figsize=(12, 10))

        # 热力图设置
        cmap = LinearSegmentedColormap.from_list('mycmap',
                                                 [(0, (1, 0, 0, 0)), (0.5, (1, 0.5, 0, 0.8)), (0.75, (1, 1, 0, 0.8)),
                                                  (0.875, (1, 1, 1, 1)), (1, (1, 1, 1, 1))])

        hmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
        extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]

        logheatmap = np.log(hmap)
        logheatmap[np.isneginf(logheatmap)] = 0
        logheatmap = ndimage.filters.gaussian_filter(logheatmap, smoothing, mode='nearest')

        ax.imshow(logheatmap, cmap=cmap, extent=extent, vmin=0, vmax=vmax)
        if title:
            ax.set_title(title, size=25)
            plt.savefig(title + '.png', bbox_inches='tight')
        if show:
            plt.show()

    @staticmethod
    def get_time(x):
        """
        x转化为XXhXXm格式
        :param x:
        :return:
        """
        return dt.datetime.strptime(str(x), '%H').strftime('%Hh%M')

    @staticmethod
    def create_gif(imagenames, gif_name):
        imagelist = [name + '.png' for name in imagenames]
        frames = []
        for name in imagelist:
            frames.append(imageio.imread(name))
        imageio.mimsave(gif_name, frames, 'GIF', duration=1)

    def heatmap_gif(self, gps, area, times, gif_name, bins=200, zoom=9, smoothing=1, vmax=4):
        imagename = []
        for time in times:
            dfHour = gps[gps['datetime'].str[0:2] == str(time)]
            imagename.append(self.get_time(time))
            self.heatmap(dfHour, area, title=self.get_time(time))
        self.create_gif(imagename, gif_name)

    @staticmethod
    def fit_area(gps):
        """
        返回合适的地图范围
        :param gps:
        :return:
        """
        latMin = gps.latitude.min()
        latMax = gps.latitude.max()
        longMin = gps.longitude.min()
        longMax = gps.longitude.max()
        return [latMin, longMin, latMax, longMax]

    @staticmethod
    def trajectory(area, gps):
        """
        绘制轨迹图
        :param gps:
        :return:
        """
        print(area)
        mapZoom = smopy.Map((area[0], area[1], area[2], area[3]), z=10)
        x, y = mapZoom.to_pixels(gps.latitude, gps.longitude)

        ax = mapZoom.show_mpl()
        ax.plot(x, y, 'or', ms=2)
        print(x.min(), y.min(), x.max(), y.max())
        plt.show()

    @staticmethod
    def add_background(area, zoom, filename='sample', source='gaode'):
        """
        绘制地图
        :param area:
        :return:
        """
        gmap.getmap(area[0], area[1], area[2], area[3], zoom, filename, source)
        mapZoom = mpimg.imread('./output/sample.png')
        plt.imshow(mapZoom)
        plt.axis('off')
        plt.show()

    @staticmethod
    def calc_timestamp_for_gps(date_time):
        """
        计算GPS的时间戳
        :param date_time:
        :return:
        """
        date = dt.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
        timestamp = (date - dt.datetime(1970, 1, 1)).total_seconds()
        return timestamp

    @staticmethod
    def calc_distance(pointA, pointB):
        """
        计算AB点距离
        :param pointA: tuple(latitude, longitude)
        :param pointB: tuple(latitude, longitude)
        :return: distance between A and B
        """
        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError('Only tuples are supported as arguments!')

        return vincenty(pointA, pointB).meters

    @staticmethod
    def calc_bearing(pointA, pointB):
        """
        计算两个点的方位角
        :param pointA: tuple(latitude, longitude)
        :param pointB: tuple(latitude, longitude)
        :return: bearing
        """
        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError('Only tuples are supported as arguments!')\

        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])

        diffLong = math.radians(pointB[1] - pointA[1])

        # θ = atan2(sin(Δlong)*cos(lat2),cos(lat1)*sin(lat2) − sin(lat1)*cos(lat2)*cos(Δlong))
        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

        initial_bearing = math.degrees(math.atan2(x, y))
        bearing = (initial_bearing + 360) % 360

        return bearing

    def add_timestamp(self):
        """
        给GPS数据绑定时间戳
        :return:
        """
        datetime = self.data['date'] + ' ' + self.data['time']
        self.data['timestamp'] = list(map(self.calc_timestamp_for_gps, datetime))

    def calc_segment_features(self):
        """
        计算参数
        :param segment_gps_features:
        :return:
        """
        self.data['time_delta'] = 0
        self.data['distance_delta'] = 0
        self.data['velocity'] = 0
        self.data['velocity_ratio'] = 0
        self.data['acceleration'] = 0
        self.data['acceleration_ratio'] = 0
        self.data['bearing_delta'] = 0
        self.data['bearing_delta_redirect'] = 0
        self.add_timestamp()

        # [分段ID,交通方式，经度，纬度，日期，时间，时间戳，时间间隔，距离间隔，速度，速度变化率，加速度，加速度变化率，方位角变化，方位角变化率]
        csv_headers = ['segment_ID', 'trans_mode', 'latitude', 'longitude', 'date', 'time', 'former_trans_mode',
                       'timestamp', 'time_delta', 'distance_delta', 'velocity', 'velocity_ratio', 'acceleration',
                       'acceleration_ratio', 'bearing_delta', 'bearing_delta_redirect']

        segment_IDs = pd.unique(self.data.segment_ID.ravel())

        cnt = 0

        for segment_ID in segment_IDs:
            cnt += 1

            # 获取当前segment的GPS轨迹点数据
            pd_segment = self.data.loc[self.data.segment_ID == segment_ID]

            # 当前segment开始的序号
            first_segment_index = pd_segment.index[0]
            print('Segment', str(cnt), ': ', segment_ID, 'starting at index ', first_segment_index, 'end at',
                  first_segment_index + len(pd_segment)-1)

            # 计算特征值
            distance_delta = []
            time_delta = []
            velocity = []
            velocity_ratio = []
            acceleration = []
            acceleration_ratio = []
            bearing_delta = []
            bearing_delta_redirect = []

            # 上一个点的数据
            pre_latitude = 0
            pre_longitude = 0
            pre_time = 0
            pre_velocity = 0
            pre_acceleration = 0
            pre_bearing = 0

            for i, row in pd_segment.iterrows():

                # 计算初始值
                if i == first_segment_index:
                    pre_latitude = row['latitude']
                    pre_longitude = row['longitude']
                    distance_delta.append(0)

                    pre_time = row['timestamp']
                    time_delta.append(0)

                    pre_velocity = 0
                    velocity.append(0)
                    velocity_ratio.append(0)

                    pre_acceleration = 0
                    acceleration.append(0)
                    acceleration_ratio.append(0)

                    pre_bearing = 0
                    bearing_delta.append(0)
                    bearing_delta_redirect.append(0)

                    continue

                # 1.计算时间间隔
                t_delta = row['timestamp'] - pre_time

                # 如果两个GPS有相同时间戳，则设置为间隔1s
                if t_delta == 0:
                    t_delta = 1
                    print('Delta time adjusted to 1 second because two gps points have the same timestamp at',
                          row['time'])

                time_delta.append(t_delta)

                # 2.计算距离
                pointA = (pre_latitude, pre_longitude)
                pointB = (row['latitude'], row['longitude'])

                d_delta = self.calc_distance(pointA, pointB)
                distance_delta.append(d_delta)

                # 3.计算速度
                v = d_delta / float(t_delta)
                velocity.append(v)

                # 4.计算速度变化率
                if pre_velocity != 0:
                    v_ratio = abs(v - pre_velocity) / float(pre_velocity)
                else:
                    v_ratio = 0
                velocity_ratio.append(v_ratio)

                # 5.计算加速度
                acc = (v - pre_velocity) / float(t_delta)
                acceleration.append(acc)

                # 6.计算加速度变化率
                if pre_acceleration != 0:
                    acc_ratio = (acc - pre_acceleration) / float(t_delta)
                else:
                    acc_ratio = 0
                acceleration_ratio.append(acc_ratio)

                # 7.计算方位角差
                bear_delta = self.calc_bearing(pointA, pointB)
                bearing_delta.append(bear_delta)

                # 8.计算方位角差变化率
                if pre_bearing != 0:
                    bear_delta_ratio = abs(bear_delta - pre_bearing) / t_delta
                else:
                    bear_delta_ratio = 0
                bearing_delta_redirect.append(bear_delta_ratio)

                # 设置下一次计算的参数
                pre_latitude = row['latitude']
                pre_longitude = row['longitude']
                pre_time = row['timestamp']
                pre_velocity = v
                pre_acceleration = acc
                pre_bearing = bear_delta

            # 更新pd_segment数据
            pd_segment.loc[:, 'time_delta'] = time_delta
            pd_segment.loc[:, 'distance_delta'] = distance_delta
            pd_segment.loc[:, 'velocity'] = velocity
            pd_segment.loc[:, 'velocity_ratio'] = velocity_ratio
            pd_segment.loc[:, 'acceleration'] = acceleration
            pd_segment.loc[:, 'acceleration_ratio'] = acceleration_ratio
            pd_segment.loc[:, 'bearing_delta'] = bearing_delta
            pd_segment.loc[:, 'bearing_delta_redirect'] = bearing_delta_redirect

            # 更新所有数据
            self.data.loc[first_segment_index:first_segment_index+len(pd_segment)-1, :] = pd_segment

    def plot_distribution(self, feature, modes=None, lowlim=0, highlim=40, numbins=10):
        """
        绘制分布图
        :param feature: 需要绘制的特征
        :param modes:   需要绘制的模式
        :param lowlim:  最低的限值
        :param highlim: 最高的限值
        :return:
        """
        bins = np.linspace(lowlim, highlim, numbins+1)
        if modes == None:
            modes = ['walk', 'bus', 'train', 'bike']
        for mode in modes:
            targetdata = np.array(self.data[self.data.trans_mode == mode].loc[:, feature])
            targetdata = np.array(targetdata)
            targetdata = targetdata[(targetdata >= lowlim) & (targetdata <= highlim)]
            counts, edges = np.histogram(targetdata, normed=True, bins=bins)
            x = (edges[:edges.size-1] + edges[1:]) / 2
            y = counts * (edges[1] - edges[0])
            plt.plot(x, y, 'o-', label=mode)

            for x_, y_ in zip(x, y):
                plt.text(x_, y_+0.01, '%.2f' % y_, ha='center', va='bottom', fontsize=7)

        plt.xlabel(feature)
        plt.ylabel('distribution')
        plt.legend()
        plt.show()

    def wgs84_to_gcj02(self, lat, long):
        """
        wgs84坐标系转gcj02坐标系
        https://github.com/wandergis/coordTransform_py/blob/master/coordTransform_utils.py
        gcj02：高德地图，腾讯地图
        :param lat: wgs84纬度
        :param long:wgs84经度
        :return:  gcj02坐标系
        """
        dlat = self._transformlat(long - 105.0, lat - 35.0)
        dlong = self._transformlong(long - 105.0, lat - 35.0)
        radlat = lat / 180.0 * pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
        dlong = (dlong * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
        mglat = lat + dlat
        mglong = long + dlong
        return [mglat, mglong]

    @staticmethod
    def _transformlat(long, lat):
        ret = -100.0 + 2.0 * long + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * long * lat + 0.2 * math.sqrt(math.fabs(long))
        ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
                math.sin(2.0 * long * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * pi) + 40.0 *
                math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
                math.sin(lat * pi / 30.0)) * 2.0 / 3.0
        return ret

    @staticmethod
    def _transformlong(long, lat):
        ret = 300.0 + long + 2.0 * lat + 0.1 * long * long + \
              0.1 * long * lat + 0.1 * math.sqrt(math.fabs(long))
        ret += (20.0 * math.sin(6.0 * long * pi) + 20.0 *
                math.sin(2.0 * long * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(long * pi) + 40.0 *
                math.sin(long / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(long / 12.0 * pi) + 300.0 *
                math.sin(long / 30.0 * pi)) * 2.0 / 3.0
        return ret

    def trans_mode_counts(self, modes=None, time=None, gif=None):
        if not modes:
            self.data['trans_mode'].value_counts().plot(kind='bar')
        else:
            return


'''
if __name__ == '__main__':
    segment = r'D:\Zhejiang University\Graduate Project\TrainingData\Data\segment_master.csv'
    vs = Visualization(segment)
    # 限定范围
    latMin = 39.64
    latMax = 40.51
    longMin = 115.76
    longMax = 116.88
    area = [latMin, longMin, latMax, longMax]

    ZJG = [30.311079, 120.079777, 30.295036, 120.092867]
    # vs.heatmap(vs.data, vs.fitarea(vs.data))
    # vs.calc_segment_features()
    vs.plot_distribution('velocity')
'''