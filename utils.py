# -*- coding=utf-8 -*-
"""
Module:     utils
Summary:    存放一些常用函数
Author:     Yuhao Jiang
Created:    2018/05/10  Ver.1.0
"""
import re
import xlwt
import pandas as pd
import math
from math import pi
from geopy.distance import vincenty

x_pi = 3.14159265358979324 * 3000.0 / 180.0
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方


def get_trip_id(user_id, num):
    num = str(num)
    return user_id + '0' * (4 - len(num)) + num


def get_position(pos, source):
    pattern = re.compile(r'.*?\[ (\d+\.\d+), (\d+\.\d+) \]')
    # GPS经纬度
    gps_data = pattern.findall(pos)
    if gps_data == []:       # 无有效GPS数据
        gps = ('0', '0')
    else:
        if source == 'Android':
            gps = gps_data[0]
        else:
            gps = (gps_data[0][1], gps_data[0][0])

    # 名义信息
    if 'describe: ' in pos:
        actual_pos = pos.split("describe: '")[1].split("' }")[0]
    else:
        actual_pos = ''

    return gps, actual_pos


def get_duration(time):
    """
    获取持续时间，单位为s
    :param time: 字符串格式的持续时间，例如'1:0:0'
    :return:
    """
    hour, minute, second = time.split(':')
    hour = int(hour)
    minute = int(minute)
    second = int(second)
    total_seconds = hour * 60 * 60 + minute * 60 + second
    return total_seconds


def csv2excel(csv_path, excel_path):
    """
    将csv格式数据保存到excel格式文件里
    :param csv_path: csv文件路径
    :param excel_path: excel文件路径
    :return:
    """
    data = pd.read_csv(csv_path)
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)

    # 创建sheet对象
    sheet = book.add_sheet('Main', cell_overwrite_ok=True)

    # 列表名
    columns = data.columns[1:]
    for i in range(len(columns)):
        sheet.write(0, i, columns[i])

    # 填入数据
    for i in range(data.shape[0]):
        for j in range(len(columns)):
            sheet.write(i+1, j, str(data.iloc[i, j+1]))

    book.save(excel_path)


def delete_user(users, not_wanted):
    """
    剔除不要的用户
    :param users: 原始用户
    :param not_wanted: 不需要的名单
    :return:
    """
    for user in not_wanted:
        users.remove(user)

    return users


def calculate_distance(start_gps, end_gps, source):
    """
    高德地图GCJ-02坐标系， 百度地图BD-09坐标系
    https://github.com/wandergis/coordTransform_py/blob/master/coordTransform_utils.py
    :param start_gps: 出发点GPS
    :param end_gps: 到达点GPS
    :param source: 判断什么坐标系
    :return: 距离
    """
    if source == 'iOS':     # 高德坐标系
        A_long, A_lat = gcj02_to_wgs84(start_gps[0], start_gps[1])
        B_long, B_lat = gcj02_to_wgs84(end_gps[0], end_gps[1])
    else:                   # 百度坐标系
        A_long, A_lat = bd09_to_wgs84(start_gps[0], start_gps[1])
        B_long, B_lat = bd09_to_wgs84(end_gps[0], end_gps[1])
    pointA = (A_lat, A_long)
    pointB = (B_lat, B_long)
    return vincenty(pointA, pointB).meters


def _transform_long(long, lat):
    """
    初步转换
    :param long: 经度
    :param lat: 纬度
    :return:
    """
    ret = 300.0 + long + 2.0 * lat + 0.1 * long * long + 0.1 * long * lat + 0.1 * math.sqrt(math.fabs(long))
    ret += (20.0 * math.sin(6.0 * long * pi) + 20.0 * math.sin(2.0 * long * pi)) * 2.0 / 3.0
    ret += (20 * math.sin(long * pi) + 40.0 * math.sin(long / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(long / 12.0 * pi) + 300.0 * math.sin(long / 30.0 * pi)) * 2.0 / 3.0
    return ret


def _transform_lat(long, lat):
    """
    初步转换
    :param long: 经度
    :param lat: 纬度
    :return:
    """
    ret = -100.0 + 2.0 * long + 3.0 * lat + 0.2 * lat * lat + 0.1 * long * lat + 0.2 * math.sqrt(math.fabs(long))
    ret += (20.0 * math.sin(6.0 * long * pi) + 20.0 * math.sin(2.0 * long * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 * math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 * math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(long, lat):
    """
    gcj坐标系转wgs坐标系
    :param long: 经度
    :param lat: 纬度
    :return:
    """
    dlong = _transform_long(long - 105.0, lat - 35.0)
    dlat = _transform_lat(long - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlong = (dlong - 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)

    wglong = long + dlong
    wglat = lat + dlat
    return long * 2 - wglong, lat * 2 - wglat


def bd09_to_gcj02(long, lat):
    """
    bd坐标系转gcj坐标系
    :param long: 经度
    :param lat: 纬度
    :return:
    """
    x = long - 0.0065
    y = lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gcjlong = z * math.cos(theta)
    gcjlat = z * math.sin(theta)
    return gcjlong, gcjlat


def bd09_to_wgs84(long, lat):
    """
    bd坐标系转wgs84坐标系
    :param long: 经度
    :param lat: 纬度
    :return:
    """
    long, lat = bd09_to_gcj02(long, lat)
    return gcj02_to_wgs84(long, lat)