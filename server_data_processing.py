# -*- coding=utf-8 -*-
"""
Module:     server_data_processing
Summary:    处理服务器后端数据，将数据转化为指定格式，与纸笔问卷相对应
Author:     Yuhao Jiang
Created:    2018/05/10  Ver.1.0
"""
import pandas as pd
import datetime as dt
import re
import numpy as np
import xlrd
import xlwt
from utils import *

raw_data_path = r'D:\Zhejiang University\Graduate Project\Data\Test\rawdata.txt'
person_path = r'D:\Zhejiang University\Graduate Project\Data\Data\Person.csv'
"""
输入数据格式:
时间	用户ID	第几次出行	出行段	出行目的	出发地点	到达地点	出发时	出发分	到达时	到达分	交通方式总数	交通方式编号	用时	出行距离	停车位置	近似距离	最终距离
输出数据格式：columns
"""
columns = ['用户ID', '星期', '日期', 'Trip_ID', '出行目的', '第几次出行',
           '出发地点经纬度', '出发地点名义信息', '到达地点经纬度', '到达地点名义信息', '出行距离',
           '出发时间', '达到时间', '出发时', '出发分', '到达时', '到达分', '出行时间',
           '出行方式总数', '主要出行方式', '出行方式1', '用时1', '出行方式2', '用时2', '出行方式3', '用时3', '出行方式4', '用时4',
           '性别', '年龄', '宿舍区', '年级', '专业大类', '自行车保有', '电动车保有', '汽车保有',
           '校内主要出行方式', '校外主要出行方式', 'Source']
data = pd.DataFrame(columns=columns)
person = pd.read_csv(person_path)
persons = person['用户ID'].ravel()

# 从txt读入数据
f = open(raw_data_path, 'r', encoding='utf-8')
month = {
    'Mar': '03',
    'Apr': '04',
    'May': '05',
}
lines = f.readlines()

lines_num = len(lines)
i = 0
num = 0
while i < lines_num:
    # 主要出行方式
    main_modes = np.zeros(11)
    # 子出行默认为None，根据实际情况添加
    subtrip = [None, None, None, None, None, None, None, None]
    raw_data = lines[i].split('\t')
    """
    rawdata范例：
    ['Tue Apr 10 17:48:16 GMT+08:00 2018', '352575073151131', '2', '0', '3', 
    "{ data: [ 30.307818023824197, 120.08914399999992 ], 
    describe: '浙江省杭州市西湖区紫金港路隧道浙江大学紫金港校区内,浙江大学(紫金港校区)建筑工程学院-B楼西70米' }",
    "{ data: [ 30.30965790239354, 120.09160345180433 ], 
    describe: '浙江省杭州市西湖区宜山路浙江大学紫金港校区内,浙江大学(紫金港校区)西教学区1栋-107号楼西北124米' }", 
    '17', '48', '17', '55', '', '1', '0:7:41', '557.87330246857', '', '312.35329520504894', '557.87330246857\n']
    """
    # 保存所有有效数据

    # 出行段为空时，这一行数据是Trip数据而不是SubTrip数据
    # 用户ID， Trip_ID , 出行目的， 出行距离, 出行方式总数
    user_id = raw_data[1]
    trip_id = get_trip_id(user_id, raw_data[2])
    trip_num = raw_data[2]
    purpose = raw_data[4]
    distance = raw_data[-4]

    # 日期，星期
    date_time = raw_data[0].split(' ')
    """
    Android: 'Tue Apr 10 17:48:16 GMT+08:00 2018'
    iOS: 'Sun Apr 08 2018 15:09:35 GMT+8'
    """
    if date_time[-1] == '2018':     # Android
        weekday = date_time[0]
        date = date_time[-1] + '/' + month[date_time[1]] + '/' + date_time[2]
        source = 'Android'
    else:                           # iOS
        weekday = date_time[0]
        date = date_time[3] + '/' + month[date_time[1]] + '/' + date_time[2]
        source = 'iOS'

    # 出发时间，到达时间
    start_time = raw_data[7] + ':' + raw_data[8]
    start_hour = raw_data[7]
    start_minute = raw_data[8]
    end_time = raw_data[9] + ':' + raw_data[10]
    end_hour = raw_data[9]
    end_minute = raw_data[10]
    duration = str(get_duration(raw_data[13]))

    # 出发位置，到达位置
    start_gps, start_pos = get_position(raw_data[5], source)
    end_gps, end_pos = get_position(raw_data[6], source)

    # 处理子出行，共有modes_num个子出行
    i += 1
    modes_num = 0
    raw_data = lines[i].split('\t')
    while raw_data[3] != '':
        modes_num += 1
        # 超过四个就舍弃
        if modes_num >= 5:
            i += 1
            raw_data = lines[i].split('\t')
            continue

        subtrip[2 * modes_num - 2] = raw_data[12]       # trans_mode
        subtrip[2 * modes_num - 1] = str(get_duration(raw_data[13]))     # subtrip_duration
        main_modes[int(raw_data[12])] += get_duration(raw_data[13])
        i += 1
        # 数据读取完毕
        if i >= lines_num - 1:
            break
        raw_data = lines[i].split('\t')

    if modes_num >= 5:
        continue
    main_mode = str(main_modes.argmax())
    # 添加个人信息
    sex = None
    age = None
    grade = None
    major = None
    area = None
    bike = None
    car = None
    ebike = None
    main_mode_in_college = None
    main_mode_out_college = None
    if user_id in persons:
        info = person[person['用户ID'] == user_id]
        index = info.index[0]
        sex = info['性别'][index]
        age = info['年龄'][index]
        grade = info['年级'][index]
        major = info['专业'][index]
        area = info['宿舍区'][index]
        bike = info['自行车保有'][index]
        car = info['汽车保有'][index]
        ebike = info['电瓶车保有'][index]
        main_mode_in_college = info['校内主要出行方式'][index]
        main_mode_out_college = info['校外主要出行方式'][index]

    line = [user_id, weekday, date, trip_id, purpose, trip_num,
            start_gps, start_pos, end_gps, end_pos, distance,
            start_time, end_time, start_hour, start_minute, end_hour, end_minute, duration,
            modes_num, main_mode,
            subtrip[0], subtrip[1], subtrip[2], subtrip[3], subtrip[4], subtrip[5], subtrip[6], subtrip[7],
            sex, age, area, grade, major, bike, ebike, car,
            main_mode_in_college, main_mode_out_college, source]
    row = []
    for l in line:
        if not isinstance(l, str):
            l = str(l)
        row.append(l)
    """
    columns = ['用户ID', '星期', '日期', 'Trip_ID', '出行目的', '第几次出行',
               '出发地点经纬度', '出发地点名义信息', '到达地点经纬度', '到达地点名义信息', '出行距离',
               '出发时间', '达到时间', '出发时', '出发分', '到达时', '到达分', '出行时间',
               '出行方式总数', '主要出行方式', '出行方式1', '用时1', '出行方式2', '用时2', '出行方式3', '用时3', '出行方式4', '用时4',
               '性别', '年龄', '宿舍区', '年级', '专业大类', '自行车保有', '电动车保有', '汽车保有',
               '校内主要出行方式', '校外主要出行方式', 'Source]
    """
    data.loc[num, :] = row
    num += 1


trip_path = r'D:\Zhejiang University\Graduate Project\Data\Data\trip.csv'
excel_path = r'D:\Zhejiang University\Graduate Project\Data\Data\trip.xls'
data.to_csv(trip_path)
csv2excel(trip_path, excel_path)