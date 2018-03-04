# -*- coding: utf-8 -*-
"""
Module:     CACL_GPS_DATA
Summary:    处理GPS数据，获得速度，加速度，方向角等参数
Author:     Yuhao Jiang
Created:    2018/01/29  Ver.1.0
"""
import pandas as pd
import datetime as dt
import math
import csv
from geopy.distance import vincenty

class CalcGpsData(object):
    def __init__(self, segment_gps_label):
        self.sg_gps = pd.read_csv(segment_gps_label, sep=',')

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
        if (type(pointA) != tuple) or (tpye(pointB) != tuple):
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
        if (type(pointA) != tuple) or (tpye(pointB) != tuple):
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
        datetime = self.sg_gps['dates'] + ' ' + self.sg_gps['time']
        self.sg_gps['timestamp'] = list(map(self.calc_timestamp_for_gps, datetime))

    def calc_segment_features(self, segment_gps_features):
        """
        计算参数
        :param segment_gps_features:
        :return:
        """
        self.sg_gps['time_delta'] = 0
        self.sg_gps['distance_delta'] = 0
        self.sg_gps['velocity'] = 0
        self.sg_gps['velocity_ratio'] = 0
        self.sg_gps['acceleration'] = 0
        self.sg_gps['acceleration_ratio'] = 0
        self.sg_gps['bearing_delta'] = 0
        self.sg_gps['bearing_delta_redirect'] = 0
        self.add_timestamp()

        # [分段ID,交通方式，经度，纬度，日期，时间，时间戳，时间间隔，距离间隔，速度，速度变化率，加速度，加速度变化率，方位角变化，方位角变化率]
        csv_headers = ['segment_ID', 'trans_mode', 'latitude', 'longitude', 'date', 'time', 'former_trans_mode',
                       'timestamp', 'time_delta', 'distance_delta', 'velocity', 'velocity_ratio', 'acceleration',
                       'acceleration_ratio', 'bearing_delta', 'bearing_delta_redirect']

        with open(segment_gps_features, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(csv_headers)

        segment_IDs = pd.unique(self.sg_gps.segment_ID.ravel())

        cnt = 0

        for segment_ID in segment_IDs:
            cnt += 1

            # 获取当前segment的GPS轨迹点数据
            pd_segment = self.sg_gps[self.sg_gps.segment_ID == segment_ID]

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
            self.sg_gps.loc[first_segment_index:first_segment_index+len(pd_segment)-1, :] = pd_segment

            with open(segment_gps_features, 'a') as f:
                (self.sg_gps[first_segment_index:first_segment_index+len(pd_segment)-1]).to_csv(
                    f, header=False, index=False)