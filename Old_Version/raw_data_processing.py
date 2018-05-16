# -*- coding: utf-8 -*-
"""
Module:     Raw_data_processing
Summary:    处理原始数据，获得距离，速度，加速度等参数
Author:     Yuhao Jiang
Created:    2017/12/06  Ver.1.0
Update:     2017/12/19  Ver.1.1 增加former_trans_mode
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
import csv

class LabelProcessing(object):
    """
    用于处理标注
    """

    def __init__(self):

        # 创建空的pandas数据用来存储标注
        columns = ['start_time', 'end_time', 'trans_mode']
        self.df_labels = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)

    def calc_delta_time(self, start_date, end_date):
        """
        计算时间差，单位为秒
        :param start_date: 开始时间
        :param end_date:   结束时间
        :return: time in seconds
        """
        sdt = dt.datetime.strptime(start_date, '%Y/%m/%d %H:%M:%S')
        edt = dt.datetime.strptime(end_date, '%Y/%m/%d %H:%M:%S')

        return (edt-sdt).total_seconds()

    def dt_to_timestamp_since_epoch(self, dt_string, dt_format):
        """
        由系统的时间计算时间戳
        :param dt_string: 包含时间的字符串数据，'2017/12/11 16:57:12'
        :param dt_format: 时间格式，datetime.datetime(2017, 12, 11, 16, 57, 12)
        :return: 标准时间戳
        """
        newdt = dt.datetime.strptime(dt_string, dt_format)
        timestamp = (newdt - dt.datetime(1970, 1, 1)).total_seconds()
        return timestamp

    def dt_ymd_format2_timestamp_since_epoch(self, dt_string):
        """

        :param dt_string: 包含时间的字符串数据，'2017/12/11 16:57:12'
        :return: 标准时间戳
        """
        newdt = dt.datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
        timestamp = (newdt - dt.datetime(1970, 1, 1)).total_seconds()
        return timestamp

    def find_labels(self, label_dir):
        """
        循环遍历文件，找到'labels'文件，把文件放入到df_labels中
        :param label_dir: 文件夹路径
        :return: 含有labels数据的pd数据
        """
        # 在原数据结构中添加一列路径信息
        self.df_labels['directory'] = ''

        directory = list()

        # 遍历所有文件和文件夹
        for dirpath, dirnames, filenames in os.walk(label_dir):

            # 提取labels文件
            for filename in [f for f in filenames if f.startswith('labels')]:
                df_temp = pd.read_table(os.path.join(dirpath, filename), header=0)
                df_temp.columns = ['start_time', 'end_time', 'trans_mode']
                self.df_labels = self.df_labels.append(df_temp, ignore_index=True)
                for i in range(len(df_temp)):
                    directory.append(dirpath)

        # 更新数据
        self.df_labels['directory'] = directory

    def process_time_label(self):
        """
        处理df_labels里的时间数据
        :return:
        """

        # 创建新列，单次行程的持续时间
        self.df_labels['duration_seconds'] = list(map(self.calc_delta_time, self.df_labels['start_time'],
                                                 self.df_labels['end_time']))

        # 创建列表，两次交通方式之间的间隔时间
        time_between_tracks = list()

        # 上一次结束的时间 && 下一次开始的时间
        start_next_track = 0
        end_last_track = 0

        for i, row in self.df_labels.iterrows():
            if i == 0:
                row['time_between_track'] = 0
                end_last_track = row['end_time']
                time_between_tracks.append(0)
                continue
            else:
                # 下一次行程开始的时间
                start_next_track = row['start_time']

                # 计算间隔
                time_between_track = self.calc_delta_time(end_last_track, start_next_track)

                # 添加时间间隔进list中
                time_between_tracks.append(time_between_track)

                # 设置下一次的end_last_track时间，即为本次的结束时间
                end_last_track = row['end_time']

        # 在df_labels中加入time_between_tracks
        self.df_labels['time_between_tracks'] = time_between_tracks

        # 标准化开始和结束时间
        self.df_labels['end_time_stamp'] = [self.dt_to_timestamp_since_epoch(t, '%Y/%m/%d %H:%M:%S')
                                            for t in self.df_labels['end_time']]
        self.df_labels['start_time_stamp'] = [self.dt_to_timestamp_since_epoch(t, '%Y/%m/%d %H:%M:%S')
                                              for t in self.df_labels['start_time']]

        # 添加分段的ID，格式为 文件路径+开始时间+结束时间+交通方式
        self.df_labels['Trip_id'] = self.df_labels['directory'].str[-8:] + \
                                        '_' + \
                                        self.df_labels['start_time'].str[0:4] + \
                                        self.df_labels['start_time'].str[5:7] + \
                                        self.df_labels['start_time'].str[8:10] + \
                                        self.df_labels['start_time'].str[11:13] + \
                                        self.df_labels['start_time'].str[14:16] + \
                                        self.df_labels['start_time'].str[17:19] + \
                                        '_' + \
                                        self.df_labels['trans_mode']

        # 添加前一段的交通方式
        cnt = 0
        former_trans_mode = []
        former_trans = None
        for i, row in self.df_labels.iterrows():
            if cnt == 0:
                former_trans_mode.append('None')
                former_trans = row['trans_mode']
                cnt += 1
                continue
            if row['time_between_tracks'] > 500:
                former_trans_mode.append('None')
                former_trans = row['trans_mode']
                cnt = 1
                continue

            former_trans_mode.append(former_trans)
            former_trans = row['trans_mode']
            cnt += 1
        self.df_labels['former_trans_mode'] = former_trans_mode


    def save_to_csv(self, file_path):
        """
        save df_labels到指定路径
        :param file_path:
        :return:
        """
        self.df_labels.to_csv(file_path)

    def search_trajectory_data(self, file_struct_path):
        """
        从文件夹里搜索GPS数据,包括"Latitude", "Longitude", "TimeStamp"
        :param file_struct_path:
        :return:
        """

        # 创建CSV文件
        with open(file_struct_path, 'w') as csvfile:
            fts_writer = csv.writer(csvfile, delimiter=',')
            fts_writer.writerow(['directory', 'file_name', 'start_time_stamp', 'end_time_stamp'])

            # 准备数据
            total_cnt = 0
            total_gps = 0

            # D:\Zhejiang University\Graduate Project\Data\Geolife Trajectories 1.3\Data\010
            for directory in np.unique(self.df_labels.directory):
                in_directory = os.path.join(directory, 'Trajectory')
                files = os.listdir(in_directory)

                cnt = 0
                # 读取GPS数据
                # D:\Zhejiang University\Graduate Project\Data\Geolife Trajectories 1.3\Data\010\Trajectory\20070804033032.csv
                for filegps in files:

                    cnt += 1
                    total_cnt += 1

                    file_path = os.path.join(in_directory, filegps)
                    f_gps = pd.read_csv(file_path, skiprows=6, header=None)

                    # 获得第一和最后的时间戳
                    start = self.dt_to_timestamp_since_epoch(f_gps[5][0] + ' ' + f_gps[6][0], '%Y-%m-%d %H:%M:%S')
                    end = self.dt_to_timestamp_since_epoch(f_gps[5][len(f_gps)-1] + ' ' + f_gps[6][len(f_gps)-1],
                                                           '%Y-%m-%d %H:%M:%S')

                    print(str(total_cnt), str(cnt) + '\t' + directory + ':\t', filegps, start, end, 'total points: ',
                          len(f_gps))
                    total_gps += len(f_gps)

                    # 写入到CSV文件中
                    fts_writer.writerow([directory, filegps, start, end])

        print('Total GPS points to search: ', total_gps)

    def create_gps_points_master(self, file_gps_path, file_struct_path):
        """
        把search_trajectory_data()的数据整合
        :param file_gps_path:
        :param file_struct_path:
        :return:
        """
        with open(file_gps_path, 'w') as csvfile:
            seg_writer = csv.writer(csvfile, delimiter=',')
            seg_writer.writerow(['segment_ID', 'trans_mode', 'latitude', 'longitude', 'date', 'time'])

            # 打开GPS数据
            df_files_info = pd.read_csv(file_struct_path)

            # 寻找带有标记的数据，并和GPS数据整合
            for i, row_lbl in self.df_labels.iterrows():
                # 获取分段的开始和结束时间
                start_time_to_search = row_lbl['start_time_stamp']
                end_time_to_search = row_lbl['end_time_stamp']

                # 寻找相同目录的数据
                dir_searched = df_files_info[df_files_info['directory'] == row_lbl['directory']]

                f_file = np.logical_and(start_time_to_search < dir_searched.end_time_stamp,
                                        end_time_to_search > dir_searched.start_time_stamp)
                res_file = f_file[f_file == True]

                if len(res_file) == 0:
                    print('No points found for ', row_lbl['start_time'])

                elif len(res_file) == 1:
                    idx = f_file[f_file == True].index
                    in_directory = os.path.join(row_lbl['directory'], 'Trajectory')
                    file_path = os.path.join(in_directory, df_files_info.loc[idx[0], 'file_name'])
                    # 打开文件
                    f_gps = pd.read_csv(file_path, skiprows=6, header=None)

                    # 获得每一个GPS点的时间戳
                    f_gps['time_stamp_gps'] = list(map(self.dt_ymd_format2_timestamp_since_epoch,
                                                   f_gps[5] + " " + f_gps[6]))

                    # t_start < timestamp < t_end
                    f_gps['valid_gps_mask'] = np.logical_and(f_gps['time_stamp_gps'] >= start_time_to_search,
                                                             f_gps['time_stamp_gps'] <= end_time_to_search)

                    # 添加数据
                    for j, row_gps in f_gps.iterrows():

                        # 只保存有效数据
                        if row_gps['valid_gps_mask'] == True:
                            seg_writer.writerow([row_lbl['segment_ID'],
                                                 row_lbl['trans_mode'],
                                                 row_lbl['former_trans_mode'],
                                                 row_gps[0],
                                                 row_gps[1],
                                                 row_gps[5],
                                                 row_gps[6]])


# 处理数据


if __name__ == '__main__':
    lp = LabelProcessing()
    lp.find_labels('D:\Zhejiang University\Graduate Project\Data\Geolife Trajectories 1.3\Data')
    lp.process_time_label()
    lp.save_to_csv('labels_master.csv')

    lp.search_trajectory_data('file_structure_master.csv')
    lp.create_gps_points_master('gps_points_master.csv', 'file_structure_master.csv')