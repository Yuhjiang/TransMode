# -*- coding=utf-8 -*-
"""
Module:     RECOGNITION
Summary:    获取GPS数据，并计算特征值，识别
Author:     Yuhao Jiang
Created:    2018/01/30  Ver.1.0
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
from geopy.distance import vincenty
import math
import datetime as dt

# 载入模型
# model = open('transport_classifier.pkl', 'rb')

Walk_Velocity_Threshold = 2.5
Walk_Acceleration_Threshold = 1.5
Time_Threshold = 20
Distance_Threshold = 30


class Recognition(object):
    def __init__(self, filename):
        self.data = pd.read_csv(filename)

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

    def calc_gps_features(self):
        """
        计算速度，加速度
        :param
        :return:
        """
        self.data['time_delta'] = 0
        self.data['distance_delta'] = 0
        self.data['velocity'] = 0
        self.data['acceleration'] = 0
        self.add_timestamp()

        distance_delta = []
        time_delta = []
        velocity = []
        acceleration = []

        # 上一个点的数据
        pre_latitude = 0
        pre_longitude = 0
        pre_time = 0
        pre_velocity = 0

        for i, row in self.data.iterrows():
            if i == 0:
                pre_latitude = row['latitude']
                pre_longitude = row['longitude']
                pre_time = row['timestamp']
                pre_velocity = 0

                distance_delta.append(0)
                time_delta.append(0)
                velocity.append(0)
                acceleration.append(0)

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

            # 4.计算加速度
            acc = (v - pre_velocity) / float(t_delta)
            acceleration.append(acc)

            # 设置下一次计算的参数
            pre_latitude = row['latitude']
            pre_longitude = row['longitude']
            pre_time = row['timestamp']
            pre_velocity = v

        self.data.loc[:, 'time_delta'] = time_delta
        self.data.loc[:, 'distance_delta'] = distance_delta
        self.data.loc[:, 'velocity'] = velocity
        self.data.loc[:, 'acceleration'] = acceleration

    def segmentation(self):
        self.calc_gps_features()
        self.data['info'] = None
        self.data['acceleration'] = abs(self.data['acceleration'])
        self.data['velocity'] = abs(self.data['velocity'])
        sg_info = []

        # 1.初步判断
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

        # 2.如果时间跨步或距离跨度小于阈值，把这一段归为前一段
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
            if i == len(sg_info)-1:
                forward_info = sg_info[i-1]
                continue
            forward_info = sg_info[i+1]
            # 如果本次info和上次不同，说明之后可能出现需要修改的情况
            if present_info != backward_info:
                # 记录下位置
                start_index = i
            # 出现当前info和前一段不同，后一个点也不同，这是一个子段
            if present_info != backward_info and backward_info != forward_info:
                # 如果时间或距离小于阈值，归于前一段
                time_span = sum(self.data.time_delta[start_index:end_index])
                distance_span = sum(self.data.distance_delta[start_index:end_index])
                if time_span < Time_Threshold or distance_span < Distance_Threshold:
                    for j in range(start_index, end_index):
                        sg_info[j] = backward_info

        self.data['info'] = sg_info


if __name__ == '__main__':
    test = Recognition('test.csv')
    test.segmentation()
    test.data.to_csv('test_1.csv')