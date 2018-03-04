# -*- coding: utf-8 -*-
"""
Module:     Segment_feature
Summary:    计算模型训练所需要的特征值
Author:     Yuhao Jiang
Created:    2017/12/13  Ver.1.0
"""

import numpy as np
import pandas as pd

"""
行走平均速度：1.388
自行车平均速度：4.30556
汽车平均速度：3.361111
公交车平均速度：5.27778
"""
AVG_walk_speed = 1.388
AVG_bike_speed = 4.30556
AVG_car_speed = 3.361111
AVG_bus_speed = 5.27778

"""
阈值：
低速度的占比：0.15
速度变化值: 5
方位角变化：30
"""
THRESHOLD_percentage_low = 0.15
THRESHOLD_change_velocity_rate = 5
THRESHOLD_change_bearing_rate = 30

THRESHOLD_low = AVG_walk_speed * THRESHOLD_percentage_low
THRESHOLD_walk_low = AVG_walk_speed * THRESHOLD_percentage_low
THRESHOLD_bike_low = AVG_bike_speed * THRESHOLD_percentage_low
THRESHOLD_car_low = AVG_car_speed * THRESHOLD_percentage_low
THRESHOLD_bus_low = AVG_bus_speed * THRESHOLD_percentage_low


# Main
class SegmentFeature(object):
    """
    基于Gps_data_engineering，计算特征值
    """

    def __init__(self, segment_file_path):
        # 载入segment数据
        self.df_seg = pd.read_csv(segment_file_path, sep=',')

    def feature_segment(self):
        # 创建新的pandas数据文件
        seg_columns = ['segment_ID',                         # Segment_ID
                       'former_trans_mode',                  # 前一段的交通工具
                       'time_total',                         # 一段segment的时间
                       'distance_total',                     # 一段segment的距离
                       'velocity_mean_distance',             # distance_total / time_total
                       'velocity_mean_segment',              # 所有速度的和/轨迹点数
                       'velocity_top1',                      # 第一速度
                       'velocity_top2',                      # 第二速度
                       'velocity_top3',                      # 第三速度
                       'acceleration_top1',                  # 第一加速度
                       'acceleration_top2',                  # 第二加速度
                       'acceleration_top3',                  # 第三加速度
                       'velocity_low_rate_distance',         # 低于阈值的速度 / distance_total
                       'velocity_change_rate',               # 速度变化超过阈值的数量 / distance_total
                       'bearing_change_rate'                 # 方位角变化超过阈值的数量 / distance_total
                       ]

        df_seg = pd.DataFrame(columns=seg_columns)
        df_seg = df_seg.fillna(0)

        # segment特征值
        segment_ID = 0
        former_trans_mode = 0
        time_total = 0
        distance_total = 0
        velocity_mean_distance = 0
        velocity_mean_segment = 0
        velocity_top1 = 0
        velocity_top2 = 0
        velocity_top3 = 0
        acceleration_top1 = 0
        acceleration_top2 = 0
        acceleration_top3 = 0
        velocity_low_rate_distance = 0
        velocity_change_rate = 0
        bearing_change_rate = 0

        # 获得segments
        segment_ids = pd.unique(self.df_seg.segment_ID.ravel())

        cnt = 0

        for segment_id in segment_ids:
            cnt += 1
            print(str(cnt), ': ', segment_id)
            former_trans_mode = self.df_seg.former_trans_mode[self.df_seg.segment_ID == segment_id]
            former_trans_mode = pd.unique(former_trans_mode.ravel())[0]
            print('former transportation mode is: ', former_trans_mode)

            pd_segment = self.df_seg[self.df_seg.segment_ID == segment_id]
            columns = ['time_delta', 'distance_delta', 'velocity', 'velocity_ratio', 'acceleration',
                       'acceleration_ratio', 'bearing_delta', 'bearing_delta_redirect']
            pd_segment[columns] = pd_segment[columns].astype(float)

            # 剔除只含有1个GPS点的数据
            if len(pd_segment) == 1:
                print('No calculation for ', segment_id)
                continue

            first_segment_index = pd_segment.index[0]
            last_segment_index = first_segment_index + len(pd_segment) - 1

            print('first_segment_index: ', first_segment_index)
            print('last_segment_index: ', last_segment_index)

            # 总时间
            time_total = np.sum(pd_segment.time_delta)
            print('time_total: ', time_total)

            # 总距离
            distance_total = np.sum(pd_segment.distance_delta)
            print('distance_total: ', distance_total)

            # 平均速度
            velocity_mean_distance = distance_total / time_total
            print('velocity_mean_distance', velocity_mean_distance)

            # 平均速度
            velocity_mean_segment = np.sum(pd_segment.velocity) / len(pd_segment)
            print('velocity_mean_segment', velocity_mean_segment)

            # 最大速度
            vd_copy = pd_segment.velocity.copy()
            topvels = sorted(vd_copy, reverse=True)

            # Top1
            velocity_top1 = topvels[0]

            # Top2 & Top3
            if len(topvels) >= 2:
                velocity_top2 = topvels[1]
            if len(topvels) >= 3:
                velocity_top3 = topvels[2]

            # 最大加速度
            ad_copy = pd_segment.acceleration.copy()
            topaccs = sorted(ad_copy)

            # Top1
            acceleration_top1 = topaccs[0]

            # Top2 & Top3
            if len(topaccs) >= 2:
                acceleration_top2 = topaccs[1]
            if len(topaccs) >= 3:
                acceleration_top3 = topaccs[2]

            # 低速度比率
            velocity_low_rate_distance = pd_segment.velocity[pd_segment.velocity < THRESHOLD_low].count()/distance_total

            # 高速度变化值比率
            velocity_change_rate = pd_segment.velocity_ratio[pd_segment.velocity_ratio >
                                                             THRESHOLD_change_velocity_rate].count()/distance_total
            # 大方位角比率
            bearing_change_rate = pd_segment.bearing_delta_redirect[pd_segment.bearing_delta_redirect >
                                                                    THRESHOLD_change_bearing_rate].count()/distance_total
            # 存储数据
            df_temp = pd.DataFrame([[segment_id,
                                    former_trans_mode,
                                    time_total,
                                    distance_total,
                                    velocity_mean_distance,
                                    velocity_mean_segment,
                                    velocity_top1,
                                    velocity_top2,
                                    velocity_top3,
                                    acceleration_top1,
                                    acceleration_top2,
                                    acceleration_top3,
                                    velocity_low_rate_distance,
                                    velocity_change_rate,
                                    bearing_change_rate]], columns=seg_columns)
            df_seg = df_seg.append(df_temp, ignore_index=True)
        return df_seg

    def save_to_csv(self, file_path, seg_featured):
        """
        存储数据到文件中
        :param file_path: 存储路径
        :return:
        """
        seg_featured.to_csv(file_path)


if __name__ == '__main__':
    sf = SegmentFeature('segment_master.csv')
    seg_featured = sf.feature_segment()
    sf.save_to_csv('segment_featured_master.csv', seg_featured)