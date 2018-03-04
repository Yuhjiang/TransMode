# -*- conding=utf-8 -*-
"""
Module:     GPS_DATA_PROCESSING
Summary:
            1. 整合所有带标注交通方式的数据 segment_with_label
            2. 利用segment_with_label，整合所有GPS数据，并绑定交通方式，前置交通方式 gps_data_processed
            3. 计算所有GPS点的速度，加速度，方位角等数据
Author:     Yuhao Jiang
Created:    2018/1/10
Update:     Version 1.0
"""

import pandas as pd
import numpy as np
import os
import datetime as dt
import sys
import csv


class SegmentLabel(object):
    """
    整合所有带标注交通方式的数据segment_with_label

    Sample:
    D:\Zhejiang University\Graduate Project\Data\Geolife Trajectories 1.3\Data\010\labels.txt
    Start Time	End Time	Transportation Mode
    2007/06/26 11:32:29	2007/06/26 11:40:29	bus
    2008/03/28 14:52:54	2008/03/28 15:59:59	train
    2008/03/28 16:00:00	2008/03/28 22:02:00	train
    """

    def __init__(self):
        # segment_with_label数据保存格式
        columns = ['start_time', 'end_time', 'trans_mode']
        # 初始化pandas数据
        self.sg_labels = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)

    @staticmethod
    def calc_delta_time(self, start_time, end_time):
        """
        计算时间差
        :param start_time: 开始时间 2007/06/26 11:32:29
        :param end_time: 结束时间   2007/06/26 11:40:29
        :return: 时间差
        """
        start = dt.datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
        end = dt.datetime.strptime(end_time, '%Y/%m/%d %H:%M:%S')
        delta_time = end - start
        return delta_time.total_seconds()

    @staticmethod
    def calc_timestamp_for_segment(self, date_time):
        """
        计算当前时间的时间戳，用于计算和比较时间
        :param date_time: 当前时间 2007/06/26 11:40:29
        :return: 相对于1970/01/01的时间戳
        """
        datetime = dt.datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
        standard_time = dt.datetime(1970, 1, 1)
        timestamp = datetime - standard_time
        return timestamp.total_seconds()

    @staticmethod
    def calc_timestamp_for_gps(self, date_time):
        """
        计算当前时间的时间戳，用于计算和比较时间
        :param date_time: 当前时间 2007-06-26 11:40:29
        :return: 相对于1970/01/01的时间戳
        """
        datetime = dt.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
        standard_time = dt.datetime(1970, 1, 1)
        timestamp = datetime - standard_time
        return timestamp.total_seconds()

    def find_labels(self, label_dir):
        """
        寻找含有标注数据的文件，整合数据到segment_with_label
        :param label_dir: 数据路径
        :return: self.sg_labels
        """

        # sefl.sg_labels添加路径信息
        self.sg_labels['directory'] = []

        directory = list()

        # 遍历路径内所有文件
        # dirpath 目录路径 dirnames 子目录名字 filenames 文件名字
        for dirpath, dirnames, filenames in os.walk(label_dir):

            # 提取labels
            for filename in [f for f in filenames if f.startswith('labels')]:
                sg_temp = pd.read_table(os.path.join(dirpath, filename), header=0)
                sg_temp.columns = ['start_time', 'end_time', 'trans_mode']
                self.sg_labels = self.sg_labels.append(sg_temp, ignore_index=True)
                for i in range(len(sg_temp)):
                    directory.append(dirpath)

        # 将目录信息保存
        self.sg_labels['directory'] = directory

    def combine_time_label(self, label_dir, file_path):
        """
        处理时间信息，计算每段行程开始结束时间戳，持续时间，两段行程间隔，前置交通方式
        :return:
        start_timestamp, end_timestamp, duration_time, time_between_tracks
        """

        self.find_labels(label_dir)
        # 计算开始时间，结束时间的时间戳
        self.sg_labels['start_timestamp'] = list(map(self.calc_timestamp_for_segment, self.sg_labels['start_time']))
        self.sg_labels['end_timestamp'] = list(map(self.calc_timestamp_for_segment, self.sg_labels['end_time']))

        # 计算持续时间
        self.sg_labels['duration_time'] = list(map(self.calc_delta_time,
                                                   self.sg_labels['start_time'], self.sg_labels['end_time']))

        # 计算两次行程的时间间隔，前置交通方式
        time_between_tracks = list()
        former_trans_mode = list()

        # 上次结束时间，本次开始时间
        end_time_last = 0
        start_time_present = 0
        num_this_user = 0
        last_user = '000'
        former_trans = 'None'

        for i, row in self.sg_labels.iterrows():
            # 当前用户
            user = row['directory']
            if user != last_user:
                num_this_user = 0

            # 某个用户第一次出行
            if num_this_user == 0:
                row['time_between_track'] = 0
                end_time_last = row['end_time']
                time_between_tracks.append(0)
                former_trans_mode.append('None')
                # 记录当前交通方式
                former_trans = row['trans_mode']
                num_this_user += 1
                last_user = user
                continue
            else:
                # 下次行程开始
                start_time_present = row['start_time']

                # 计算间隔
                # 本次开始的时间-上次结束的时间
                time_between_track = self.calc_delta_time(end_time_last, start_time_present)
                time_between_tracks.append(time_between_track)

                # 设置下一次的结束时间
                end_time_last = row['end_time']

                num_this_user += 1
                last_user = user

                # 间隔超过阈值，认为上次出行已经结束
                if time_between_track > 500:
                    former_trans_mode.append('None')
                else:
                    former_trans_mode.append(former_trans)

                former_trans = row['trans_mode']

        # 加入time_between_tracks
        self.sg_labels['time_between_tracks'] = time_between_tracks

        # 添加segment ID，格式为 文件路径（包括用户ID）+开始时间+交通方式
        self.sg_labels['segment_ID'] = self.sg_labels['directory'].str[-8:] + '_' + \
                                       self.sg_labels['start_time'].str[0:4] + \
                                       self.sg_labels['start_time'].str[5:7] + \
                                       self.sg_labels['start_time'].str[8:10] + \
                                       self.sg_labels['start_time'].str[11:13] + \
                                       self.sg_labels['start_time'].str[14:16] + \
                                       self.sg_labels['start_time'].str[17:19] + \
                                       '_' + self.sg_labels['trans_mode']

        # 添加前置交通方式
        self.sg_labels['former_trans_mode'] = former_trans_mode

        # 保存文件
        self.sg_labels.to_csv(file_path)

    def find_all_gps(self, all_gps_path):
        """
        把有标注的用户的GPS用户所有GPS数据找出来，并计算时间
        :param all_gps_path:
        :return:
        """
        columns = ['directory', 'file_name', 'start_timestamp', 'end_timestamp']
        with open(all_gps_path, 'w') as gpsfile:
            fts_writer = csv.writer(gpsfile, delimiter=',')
            fts_writer.writerow(columns)

            total_cnt = 0
            total_gps = 0

            # D:\Zhejiang University\Graduate Project\Data\Geolife Trajectories 1.3\Data\010
            for directory in np.unique(self.sg_labels.directory):
                gps_directory = os.path.join(directory, 'Trajectory')
                files = os.listdir(gps_directory)

                cnt = 0
                # 读取GPS数据
                # D:\Zhejiang University\Graduate Project\Data\Geolife Trajectories 1.3\Data\010\Trajectory\20070804033032.csv
                for gps_file in files:
                    cnt += 1
                    total_cnt += 1

                    file_path = os.path.join(gps_directory, gps_file)
                    f_gps = pd.read_csv(file_path, skiprows=6, header=None)

                    # 获得第一个和最后的时间戳
                    start_time = self.calc_timestamp_for_gps(f_gps[5][0] + ' ' + f_gps[6][0])
                    end_time = self.calc_timestamp_for_gps(f_gps[5][len(f_gps)-1] + ' ' + f_gps[6][len(f_gps)-1])

                    print(str(total_cnt), str(cnt), '\t', directory, ':\t', gps_file, start_time, end_time, 'total points: ',
                          len(f_gps))
                    total_gps += len(f_gps)
                    fts_writer.writerow([directory, gps_file, start_time, end_time])
        print('Total Gps Points to search: ', total_gps)

    def combine_gps_segment(self, all_gps_path, combine_gps_segment):
        """
        汇总所有GPS数据
        :param all_gps_path: GPS数据时间，目录汇总
        :param combine_gps_segment: 保存所有GPS数据
        :return:
        """
        columns = ['segment_ID', 'trans_mode', 'former_trans_mode' 'latitude', 'longitude', 'date', 'time']
        with open(combine_gps_segment, 'w') as csvfile:
            seg_writer = csv.writer(csvfile, delimiter=',')
            seg_writer.writerow(columns)

            # 打开all_gps_path
            gps_files_info = pd.read_csv(all_gps_path)

            # 寻找带有标记的数据，并和GPS整合在一起
            for i, row in self.sg_labels.iterrows():
                # 获取segment的开始和结束时间
                start_time_to_search = row['start_timestamp']
                end_time_to_search = row['end_timestamp']

                # 寻找同一个目录下的数据
                dir_to_search = gps_files_info[gps_files_info['directory'] == row['directory']]

                gps_file = np.logical_and(start_time_to_search < dir_to_search.end_timestamp,
                                          end_time_to_search > dir_to_search.start_timestamp)

                gps_searched = gps_file[gps_file == True]

                if len(gps_searched) == 0:
                    print('No points found for ', row['start_time'])

                elif len(gps_searched) == 1:
                    index = gps_searched.index
                    in_directory = os.path.join(row['directory'], 'Trajectory')
                    gps_path = os.path.join(in_directory, gps_files_info.loc[index[0], 'file_name'])
                    # 打开文件
                    f_gps = pd.read_csv(gps_path, skiprows=6, header=None)

                    # 时间戳
                    f_gps['time_stamp_gps'] = list(map(self.calc_timestamp_for_gps, f_gps[5] + ' ' + f_gps[6]))

                    # 截取有效的GPS数据
                    f_gps['valid_gps_mask'] = np.logical_and(f_gps['time_stamp_gps'] >= start_time_to_search,
                                                             f_gps['time_stamp_gps'] <= end_time_to_search)

                    # 把有效数据加入文件中
                    for j, row_gps in f_gps.iterrows():
                        if row_gps['valid_gps_mask']:
                            # 'segment_ID', 'trans_mode', 'former_trans_mode', 'latitude', 'longitude', 'date', 'time'
                            seg_writer.writerow([row['segment_ID'], row['trans_mode'], row['former_trans_mode'],
                                                 row_gps[0], row_gps[1], row_gps[5], row_gps[6]])

    def get_segment_gps(self, label_dir, segment_gps_label):
        """
        整合模块功能
        :param label_dir:
        :param segment_gps_label:
        :return:
        """
        self.combine_time_label(label_dir, 'segment_time_label.csv')
        self.find_all_gps('segment_gps.csv')
        self.combine_gps_segment('segment_gps.csv', segment_gps_label)


"""
if __name__ == '__main__':
    lp = SegmentLabel()
    lp.combine_time_label('D:\Zhejiang University\Graduate Project\Data\Geolife Trajectories 1.3\Data',
                          'segment_time_label.csv')
    lp.find_all_gps('segment_gps.csv')

    lp.combine_gps_segment('segment_gps.csv', 'segment_gps_label.csv')
"""