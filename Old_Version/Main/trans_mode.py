# -*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import pickle
from geopy.distance import vincenty
import math
import datetime as dt
import sys
import json
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing

# 载入模型
model = open(r'.\transport_classifier.pkl', 'rb')

Walk_Velocity_Threshold = 2.5
Walk_Acceleration_Threshold = 1.5
Time_Threshold = 30
Distance_Threshold = 50

# 计算特征的参数
Average_Walk_Velocity = 1.388
Low_Threshold_Percentage = 0.15
Change_Velocity_Rate_Threshold = 5
Change_Bearing_Rate_Threshold = 30

Low_Threshold = Average_Walk_Velocity * Low_Threshold_Percentage


class Recognition(object):
    def __init__(self, raw_data):
        self.data = pd.read_csv(raw_data)
        #self.data = self.json2pandas()
        self.rf = pickle.load(model)

    @staticmethod
    def calc_distance(pointA, pointB):
        """
        计算A，B点距离
        :param pointA:
        :param pointB:
        :return:
        """
        return vincenty(pointA, pointB).meters

    @staticmethod
    def calc_timestamp(date_time):
        """
        计算时间戳
        :param date_time:
        :return:
        """
        date = dt.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
        timestamp = (date - dt.datetime(1970, 1, 1)).total_seconds()
        return timestamp

    def add_timestamp(self):
        self.data['timestamp'] = list(map(self.calc_timestamp, self.data['date'] + ' ' + self.data['time']))

    @staticmethod
    def calculate_initial_compass_bearing(pointA, pointB):
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

    def json2pandas(self):
        """
        把输入的list数据转化成pandas数据
        :return:
        """
        Trip_ID = []
        trans_modes = []
        former_trans_modes = []
        latitudes = []
        longitudes = []
        dates = []
        times = []

        for i in range(0, len(self.raw_data)):
            row_i = self.raw_data[i]
            for j in range(0, len(row_i['data'])):
                row_j = self.raw_data[i]['data'][j]
                Trip_ID.append(row_i['Trip_ID'])
                trans_modes.append(row_i['trans_mode'])
                former_trans_modes.append(row_i['former_trans_mode'])
                latitudes.append(float(row_j['latitude']))
                longitudes.append(float(row_j['longitude']))
                dates.append(row_j['date'])
                times.append(row_j['time'])

        d = {
            'Trip_ID': Trip_ID,
            'trans_mode': trans_modes,
            'former_trans_mode': former_trans_modes,
            'latitude': latitudes,
            'longitude': longitudes,
            'date': dates,
            'time': times
        }
        # columns = ['Trip_ID', 'trans_mode', 'former_trans_mode', 'latitude', 'longitude', 'date', 'time']
        # array = np.array([Trip_IDs, trans_modes, former_trans_modes, latitudes, longitudes, dates, times]).transpose()
        # data = pd.DataFrame(array, columns=columns)
        data = pd.DataFrame(d)
        return data

    def calc_gps_feature(self):
        """
        计算速度，加速度
        :param
        :return:
        """
        # 增加新的列
        self.data['time_delta'] = 0
        self.data['distance_delta'] = 0
        self.data['velocity'] = 0
        self.data['velocity_ratio'] = 0
        self.data['acceleration'] = 0
        self.data['acceleration_ratio'] = 0
        self.data['bearing_delta'] = 0
        self.data['bearing_delta_redirect'] = 0
        self.add_timestamp()

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

        for i, row in self.data.iterrows():
            if i == 0:
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

            time_delta.append(t_delta)

            # 2.计算距离
            pointA = (pre_latitude, pre_longitude)
            pointB = (row['latitude'], row['longitude'])

            d_delta = self.calc_distance(pointA, pointB)
            distance_delta.append(d_delta)

            # 3.计算速度
            v = d_delta / float(t_delta)
            velocity.append(v)

            # 4.计算速度变化率，在加速度里再考虑角速度的符号。这里使用绝对值
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
                acc_ratio = abs((acc - pre_acceleration) / float(pre_acceleration))
            else:
                acc_ratio = 0

            acceleration_ratio.append(acc_ratio)

            # 7.计算方位角差
            bear_delta = self.calculate_initial_compass_bearing(pointA, pointB)

            bearing_delta.append(bear_delta)

            # 8.计算方位角差变化率
            if pre_bearing != 0:
                bear_delta_ratio = abs(bear_delta - pre_bearing)
            else:
                bear_delta_ratio = 0

            bearing_delta_redirect.append(bear_delta_ratio)

            # 设置当前的参数，用作下一次循环用
            pre_latitude = row['latitude']
            pre_longitude = row['longitude']
            pre_time = row['timestamp']
            pre_velocity = v
            pre_acceleration = acc
            pre_bearing = bear_delta

        self.data.loc[:, 'time_delta'] = time_delta
        self.data.loc[:, 'distance_delta'] = distance_delta
        self.data.loc[:, 'velocity'] = velocity
        self.data.loc[:, 'velocity_ratio'] = velocity_ratio
        self.data.loc[:, 'acceleration'] = acceleration
        self.data.loc[:, 'acceleration_ratio'] = acceleration_ratio
        self.data.loc[:, 'bearing_delta'] = bearing_delta
        self.data.loc[:, 'bearing_delta_redirect'] = bearing_delta_redirect

    def segmentation(self):
        self.calc_gps_feature()
        self.data['info'] = 'Possible-Walk'
        self.data['acceleration'] = abs(self.data['acceleration'])
        self.data['velocity'] = abs(self.data['velocity'])
        # Non_Walk和Possible_Walk
        sg_info = []
        sg_index = []

        # 1.初步判断
        '''
        for i, row in self.data.iterrows():
            if i == 0:
                sg_info.append('None')
            else:
                if row['velocity'] > Walk_Velocity_Threshold or row['acceleration'] > Walk_Acceleration_Threshold:
                    sg_info.append('Non-Walk')
                    if i == 1:
                        sg_info[0] = 'Non-Walk'
                else:
                    sg_info.append('Possible-Walk')
                    if i == 1:
                        sg_info[0] = 'Possible-Walk'
        '''
        self.data.loc[self.data['velocity'] > 2.5, ['info']] = 'Non-Walk'
        self.data.loc[self.data['acceleration'] > 1.5, ['info']] = 'Non-Walk'
        self.data.loc[0, ['info']] = self.data.loc[1, 'info']
        # 2.如果时间跨步或距离跨度小于阈值，把这一段归为前一段
        start_index = 0
        end_index = 0
        sg_index = list()
        backward_info = self.data['info'][0]
        sg_index.append(0)
        for i, row in self.data.iterrows():
            if row['info'] != backward_info:
                sg_index.append(i)
                backward_info = row['info']
        sg_index.append(len(self.data))
        for i in range(1, len(sg_index)-1):
            start_index = sg_index[i]
            end_index = sg_index[i+1]
            if float(self.data.loc[start_index:end_index, ['time_delta']].sum()) < Time_Threshold or \
                    float(self.data.loc[start_index:end_index, ['distance_delta']].sum()) < Distance_Threshold:
                self.data.loc[start_index:end_index, ['info']] = self.data['info'][start_index-1]
        print('what')
        '''
        time_span = 0
        distance_span = 0
        start_index = 0
        end_index = 0
        backward_info = 'None'
        forward_info = 'None'
        num = 0
        for i, row in self.data.iterrows():
            present_info = sg_info[i]
            end_index = i
            if i == 0:
                backward_info = sg_info[0]
                continue
            if i == len(sg_info) - 1:
                forward_info = sg_info[i - 1]
                continue
            forward_info = sg_info[i + 1]
            # 如果本次info和上次不同，说明之后可能出现需要修改的情况
            if present_info != backward_info:
                # 记录下位置
                start_index = i
            # 出现当前info和前一段不同，后一个点也不同，这是一个子段
            if present_info != backward_info and present_info != forward_info:
                # 如果时间或距离小于阈值，归于前一段
                time_span = sum(self.data.time_delta[start_index:end_index])
                distance_span = sum(self.data.distance_delta[start_index:end_index])
                if time_span < Time_Threshold or distance_span < Distance_Threshold:
                    for j in range(start_index, end_index):
                        sg_info[j] = backward_info
        '''

    def save_to_csv(self, filename):
        data_info = self.data.loc[:, ['trans_mode', 'time_delta', 'distance_delta', 'velocity', 'acceleration', 'info']]
        data_info.to_csv(filename)


if __name__ == '__main__':
    test = Recognition('test.csv')
    test.segmentation()
    test.data.to_csv('test_2.csv')
    test.save_to_csv('test_1.csv')