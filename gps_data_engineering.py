# -*- coding: utf-8 -*-
"""
Module:     Gps_data_engineering
Summary:    处理GPS数据，获得速度，加速度，方向角等参数
Author:     Yuhao Jiang
Created:    2017/12/12  Ver.1.0
"""
import pandas as pd
import datetime as dt
import math
import csv
from geopy.distance import vincenty

class GpsDataEngineering(object):
    """
    基于经纬度，时间戳计算参数
    """
    # gps_points_master.csv
    def __init__(self, gps_file_path):
        self.df_gps = pd.read_csv(gps_file_path, sep=',')

    def dt_to_timestamp(self, dt_string):
        """
        计算当前时间的时间戳
        :param dt_string:
        :return: timestamp
        """
        date = dt.datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
        timestamp = (date - dt.datetime(1970, 1, 1)).total_seconds()
        return timestamp

    def calculate_distance(self, pointA, pointB):
        """
        计算两个轨迹点之间的距离
        :param pointA: tuple(latitude, longitude)
        :param pointB: tuple(latitude, longitude)
        :return: distance
        """
        return vincenty(pointA, pointB).meters

    def calculate_initial_compass_bearing(self, pointA, pointB):
        """
        计算两个点的方位角
        :param pointA: tuple(latitude, longitude) (39.894178, 116.3182)
        :param pointB: tuple(latitude, longitude) (39.894505, 116.321132)
        :return: compass_bearing                  81.72820612688776
        """

        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError('Only tuples are supported as argument')

        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])

        diffLong = math.radians(pointB[1] - pointA[1])

        # 方位角（-PI to PI)
        # θ = atan2(sin(Δlong)*cos(lat2),cos(lat1)*sin(lat2) − sin(lat1)*cos(lat2)*cos(Δlong))
        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

        initial_bearing = math.atan2(x, y)

        # 标准方位角（0 to 360 degrees)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    def add_timestamp(self):
        """
        给每一个GPS点数据建立一个标准时间戳
        :return: timestamp
        """
        td = self.df_gps['date'] + ' ' + self.df_gps['time']
        self.df_gps['time_stamp'] = list(map(self.dt_to_timestamp, td))

    def calculate_segment_characteristics(self, segment_file_path):
        """
        计算参数
        :param segment_file_path: 'segment_master.csv'
        :return:
        """

        # 增加新的列
        self.df_gps['time_delta'] = 0
        self.df_gps['distance_delta'] = 0
        self.df_gps['velocity'] = 0
        self.df_gps['velocity_ratio'] = 0
        self.df_gps['acceleration'] = 0
        self.df_gps['acceleration_ratio'] = 0
        self.df_gps['bearing_delta'] = 0
        self.df_gps['bearing_delta_redirect'] = 0

        # CSV文件格式
        # [分段ID,交通方式，经度，纬度，日期，时间，时间戳，时间间隔，距离间隔，速度，速度变化率，加速度，加速度变化率，方位角变化，方位角变化率]
        csv_headers = ['segment_ID', 'trans_mode', 'latitude', 'longitude', 'date', 'time', 'former_trans_mode',
                       'time_stamp', 'time_delta', 'distance_delta', 'velocity', 'velocity_ratio', 'acceleration',
                       'acceleration_ratio', 'bearing_delta', 'bearing_delta_redirect']

        # 保存segment数据
        with open(segment_file_path, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(csv_headers)

        # 获得segment_ID
        segment_ids = pd.unique(self.df_gps.segment_ID.ravel())

        cnt = 0

        for segment_id in segment_ids:
            cnt += 1

            # 获取当前segment的gps轨迹点数据
            pd_segment = self.df_gps[self.df_gps.segment_ID == segment_id]

            # 当前segment开始的序号
            first_segment_index = pd_segment.index[0]
            print('Segment', str(cnt), ': ', segment_id, 'starting at index ', first_segment_index,
                  'end at', first_segment_index + len(pd_segment)-1)

            # 创建列表存储计算结果
            distance_delta = []
            time_delta = []
            velocity = []
            velocity_ratio = []
            acceleration = []
            acceleration_ratio = []
            bearing_delta = []
            bearing_delta_redirect = []

            # 上一个点的数据
            prev_latitude = 0
            prev_longitude = 0
            prev_time = 0
            prev_velocity = 0
            prev_acceleration = 0
            prev_bearing = 0

            for i, row in pd_segment.iterrows():

                # 计算初始值
                if i == first_segment_index:
                    prev_latitude = row['latitude']
                    prev_longitude = row['longitude']
                    distance_delta.append(0)

                    prev_time = row['time_stamp']
                    time_delta.append(0)

                    prev_velocity = 0
                    velocity.append(0)
                    velocity_ratio.append(0)

                    prev_acceleration = 0
                    acceleration.append(0)
                    acceleration_ratio.append(0)

                    prev_bearing = 0
                    bearing_delta.append(0)
                    bearing_delta_redirect.append(0)

                    continue

                # 1.计算时间间隔
                t_delta = row['time_stamp'] - prev_time

                # 如果两个GPS点有相同的时间戳，则设置时间间隔为1s
                if t_delta == 0:
                    t_delta = 1
                    print('Delta time adjusted to 1 second because two gps points have the same timestamp at',
                          row['time'])

                time_delta.append(t_delta)

                # 2.计算距离间隔
                pointA = (prev_latitude, prev_longitude)
                pointB = (row['latitude'], row['longitude'])

                d_delta = self.calculate_distance(pointA, pointB)

                distance_delta.append(d_delta)

                # 3.计算速度
                try:
                    v = d_delta / float(t_delta)
                except:
                    print('Divided by 0 at: SegId | Timestamp | Time | DDelta | TDelta',
                          segment_id, row['time_stamp'], row['time'], d_delta, t_delta)

                velocity.append(v)

                # 4.计算速度变化率，在加速度里再考虑角速度的符号。这里使用绝对值
                if prev_velocity != 0:
                    v_ratio = abs(v - prev_velocity) / float(prev_velocity)
                else:
                    v_ratio = 0

                velocity_ratio.append(v_ratio)

                # 5.计算加速度
                acc = (v - prev_velocity) / t_delta

                acceleration.append(acc)

                # 6.计算加速度变化率
                if prev_acceleration != 0:
                    acc_ratio = abs((acc - prev_acceleration) / float(prev_acceleration))
                else:
                    acc_ratio = 0

                acceleration_ratio.append(acc_ratio)

                # 7.计算方位角差
                bear_delta = self.calculate_initial_compass_bearing(pointA, pointB)

                bearing_delta.append(bear_delta)

                # 8.计算方位角差变化率
                if prev_bearing != 0:
                    bear_delta_ratio = abs(bear_delta - prev_bearing)
                else:
                    bear_delta_ratio = 0

                bearing_delta_redirect.append(bear_delta_ratio)

                # 设置当前的参数，用作下一次循环用
                prev_latitude = row['latitude']
                prev_longitude = row['longitude']
                prev_time = row['time_stamp']
                prev_velocity = v
                prev_acceleration = acc
                prev_bearing = bear_delta

            # 更新当前段的数据
            pd_segment.loc[:, 'time_delta'] = time_delta
            pd_segment.loc[:, 'distance_delta'] = distance_delta
            pd_segment.loc[:, 'velocity'] = velocity
            pd_segment.loc[:, 'velocity_ratio'] = velocity_ratio
            pd_segment.loc[:, 'acceleration'] = acceleration
            pd_segment.loc[:, 'acceleration_ratio'] = acceleration_ratio
            pd_segment.loc[:, 'bearing_delta'] = bearing_delta
            pd_segment.loc[:, 'bearing_delta_redirect'] = bearing_delta_redirect

            # 更新所有数据
            self.df_gps.loc[first_segment_index:first_segment_index+len(pd_segment)-1, :] = pd_segment

            with open(segment_file_path, 'a') as f:
                (self.df_gps[first_segment_index:first_segment_index+len(pd_segment)-1]).to_csv(
                    f, header=False, index=False)


if __name__ == '__main__':
    gpsdataengineering = GpsDataEngineering('gps_points_master.csv')
    gpsdataengineering.add_timestamp()
    gpsdataengineering.calculate_segment_characteristics('segment_master.csv')
