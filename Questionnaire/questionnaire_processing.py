# -*- coding=utf-8 -*-
"""
Module:     questionnaire_processing
Summary:    处理纸币问卷数据
Author:     Yuhao Jiang
Created:    2018/05/16  Ver.1.0
"""
import pandas as pd
import numpy as np
import xlwt
from utils import *

questionnaire_path = r'D:\Zhejiang University\Graduate Project\Data\Data\All_trip_questionary.csv'
person_path = r'D:\Zhejiang University\Graduate Project\Data\Data\Person.csv'
raw_data = pd.read_csv(questionnaire_path)
person = pd.read_csv(person_path)
persons = person['用户ID'].unique()
"""person = pd.read_csv(person_path)
数据sample:
用户ID,日期,次数,出行目的,出发地点,达到地点,出发时间,达到时间,出行方式1,用时1,出行方式2,用时2,出行方式3,用时3,出行方式4,用时4,出行路线,其他
460FAD4B-1357-45A6-8AEB-504F8E716CAD,4/17/2018,1,2,食堂,建工实验大厅,8:10:00,8:15:00,2,5, , , , , , ,6-19-30, 
"""
columns = ['用户ID', '星期', '日期', 'Trip_ID', '出行目的', '第几次出行', '出发地点', '到达地点',
           '出发时间', '达到时间', '出发时', '出发分', '到达时', '到达分', '出行时间',
           '出行方式总数', '主要出行方式', '出行方式1', '用时1', '出行方式2', '用时2', '出行方式3', '用时3', '出行方式4', '用时4',
           '路线', '其他',
           '性别', '年龄', '宿舍区', '年级', '专业大类', '自行车保有', '电动车保有', '汽车保有',
           '校内主要出行方式', '校外主要出行方式']
data = pd.DataFrame(columns=columns)

weekdays = {
    '2018/04/17': 'Thu',
    '2018/04/18': 'Web',
    '2018/04/19': 'Thu',
    '2018/04/20': 'Fri',
    '2018/04/21': 'Sat',
    '2018/04/22': 'Sun',
}

lines = raw_data.shape[0]
num = 0
for i in range(lines):
    subtrip = [None] * 8
    user_id, date_inverse, trip_num, purpose, start_pos, end_pos, starttime, endtime, subtrip[0], subtrip[1], \
        subtrip[2], subtrip[3], subtrip[4], subtrip[5], subtrip[6], subtrip[7], trajectory, other = raw_data.loc[i, :]

    # 生成星期和日期
    raw_date = date_inverse.split('/')
    date = raw_date[-1] + '/' + '0' + raw_date[0] + '/' + raw_date[1]
    weekday = weekdays[date]

    # Trip_ID
    trip_id = get_trip_id(user_id, trip_num)

    # 处理时间
    # 只需要时和分
    start_time = starttime[:-3]
    end_time = endtime[:-3]

    start_hour, start_minute = start_time.split(':')
    end_hour, end_minute = end_time.split(':')

    # 出行方式总数主出行方式
    main_modes = np.zeros(11)
    modes_num = 0
    duration = 0
    print(user_id, subtrip)
    for j in list(range(8))[::2]:
        if subtrip[j] != ' ':
            modes_num += 1
            subtrip[j+1] = int(subtrip[j+1]) * 60
            main_modes[int(subtrip[j])] += subtrip[j+1]
            duration += subtrip[j+1]
        else:
            subtrip[j] = None
            subtrip[j+1] = None

    main_mode = main_modes.argmax()

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

    line = [user_id, weekday, date, trip_id, purpose, trip_num, start_pos, end_pos,
            start_time, end_time, start_hour, start_minute, end_hour, end_minute, duration,
            modes_num, main_mode,
            subtrip[0], subtrip[1], subtrip[2], subtrip[3], subtrip[4], subtrip[5], subtrip[6], subtrip[7],
            trajectory, other,
            sex, age, area, grade, major, bike, ebike, car,
            main_mode_in_college, main_mode_out_college]
    row = []
    for l in line:
        if not isinstance(l, str):
            l = str(l)
        row.append(l)

    """
columns = ['用户ID', '星期', '日期', 'Trip_ID', '出行目的', '第几次出行', '出发地点', '到达地点',
           '出发时间', '达到时间', '出发时', '出发分', '到达时', '到达分', '出行时间',
           '出行方式总数', '主要出行方式', '出行方式1', '用时1', '出行方式2', '用时2', '出行方式3', '用时3', '出行方式4', '用时4',
           '路线', '其他',
           '性别', '年龄', '宿舍区', '年级', '专业大类', '自行车保有', '电动车保有', '汽车保有',
           '校内主要出行方式', '校外主要出行方式']
    """
    data.loc[num, :] = row
    num += 1

trip_path = r'D:\Zhejiang University\Graduate Project\Data\Data\Cleaned\trip_questionaire.csv'
excel_path = r'D:\Zhejiang University\Graduate Project\Data\Data\Cleaned\trip_questionaire.xls'
data.to_csv(trip_path)
csv2excel(trip_path, excel_path)