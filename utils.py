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