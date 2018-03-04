# -*- coding: utf-8 -*-
"""
Module:     Modeling
Summary:    建立并训练模型
Author:     Yuhao Jiang
Created:    2017/12/13  Ver.0.1
Update:     2017/12/14  Ver.1.0
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score

class ImproveModel(object):
    """
    考虑转移系数，修正模型输出结果
    """
    def __init__(self):
        self.df_seg = pd.read_csv('segment_featured_master.csv', index_col=0)
        model_data = open('transport_classifier.pkl', 'rb')
        self.rf = pickle.load(model_data)
        self.X_test_final = None
        self.y_test = None
        self.y_predict = None

    def reduce_low_sample_modes(self):
        """
        删除boat, run, airplane, train, subway等低频交通方式
        :return:
        """
        self.df_seg['trans_mode'] = (self.df_seg['segment_ID']).str[24:]

        # 剔除当前交通方式为boat, run, airplane, train, subway
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'boat']
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'run']
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'airplane']
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'train']
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'subway']
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'motorcycle']

        # 提出前置交通方式为boat, run, airplane, train, subway
        self.df_seg = self.df_seg[self.df_seg.former_trans_mode != 'boat']
        self.df_seg = self.df_seg[self.df_seg.former_trans_mode != 'run']
        self.df_seg = self.df_seg[self.df_seg.former_trans_mode != 'airplane']
        self.df_seg = self.df_seg[self.df_seg.former_trans_mode != 'train']
        self.df_seg = self.df_seg[self.df_seg.former_trans_mode != 'subway']
        self.df_seg = self.df_seg[self.df_seg.former_trans_mode != 'motorcycle']

        self.df_seg.trans_mode[self.df_seg.trans_mode == 'taxi'] = 'car'
        self.df_seg.former_trans_mode[self.df_seg.former_trans_mode == 'taxi'] = 'car'

    """                 Later Transportation Mode
            bike    bus     car    walk
    
    bike    26       14      15         0
    
    bus     14       37      40     0
    
    car      1        1      16     0
    
    walk     84      633    1004    0
    """
    def transfer_matrix(self):
        bike_bike = 0
        bike_bus = 0
        bike_car = 0
        bike_walk = 0
        bus_bike = 0
        bus_bus = 0
        bus_car = 0
        bus_walk = 0
        car_bike = 0
        car_bus = 0
        car_car = 0
        car_walk = 0
        walk_bike = 0
        walk_bus = 0
        walk_car = 0
        walk_walk = 0

        for i, row in self.df_seg.iterrows():
            former_trans = row['former_trans_mode']
            segment_trans = row['trans_mode']

            if segment_trans == 'bike':
                if former_trans == 'bike':
                    bike_bike += 1
                if former_trans == 'bus':
                    bus_bike += 1
                if former_trans == 'car':
                    car_bike += 1
                if former_trans == 'walk':
                    walk_bike += 1
            if segment_trans == 'bus':
                if former_trans == 'bike':
                    bike_bus += 1
                if former_trans == 'bus':
                    bus_bus += 1
                if former_trans == 'car':
                    car_bus += 1
                if former_trans == 'walk':
                    walk_bus += 1
            if segment_trans == 'car':
                if former_trans == 'bike':
                    bike_car += 1
                if former_trans == 'bus':
                    bus_car += 1
                if former_trans == 'car':
                    car_car += 1
                if former_trans == 'walk':
                    walk_car += 1
            if segment_trans == 'bus':
                if former_trans == 'bike':
                    bike_car += 1
                if former_trans == 'bus':
                    bus_car += 1
                if former_trans == 'car':
                    car_car += 1
                if former_trans == 'walk':
                    walk_car += 1

        transfer_m = np.array([[bike_bike, bike_bus, bike_car, bike_walk],
                              [bus_bike,  bus_bus,  bus_car,  bus_walk],
                              [car_bike,  car_bus,  car_car,  car_walk],
                              [walk_bike, walk_bus, walk_car, walk_walk]])
        print('--------------------------------------------------------')
        print(transfer_m)
        print('--------------------------------------------------------')
        # 计算转移概率
        transfer_probability = np.zeros([4, 4])
        for i in range(0, 4):
            for j in range(0, 4):
                transfer_probability[i][j] = transfer_m[i][j] / float(np.sum(transfer_m[i]))

        print('--------------------------------------------------------')
        print(transfer_probability)
        print('--------------------------------------------------------')

        return transfer_probability

    def get_probability(self):
        """
        输出概率值结果
        :return:
        """
        self.reduce_low_sample_modes()

        feature_cols = ['time_total',
                        'distance_total',
                        'velocity_mean_distance',
                        'velocity_mean_segment',
                        'velocity_top1',
                        'velocity_top2',
                        'velocity_top3',
                        'acceleration_top1',
                        'acceleration_top2',
                        'acceleration_top3',
                        'velocity_low_rate_distance',
                        'velocity_change_rate',
                        'bearing_change_rate'
                        ]

        self.X_test_final = self.df_seg[feature_cols]
        self.y_test = self.df_seg['trans_mode']
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(np.unique(self.y_test))
        self.y_test = label_encoder.transform(self.y_test)
        # bike bus car bike
        y_predict_probability = self.rf.predict_proba(self.X_test_final)
        tp = self.transfer_matrix()
        cnt = 0
        for i in self.df_seg.index:
            fomer_trans = self.df_seg.former_trans_mode[i]
            if self.df_seg.former_trans_mode[i] is 'None':
                cnt += 1
                continue
            else:
                if self.df_seg.former_trans_mode[i] is 'bike':
                    for j in range(0, 4):
                        y_predict_probability[cnt][j] = tp[0][j] * y_predict_probability[cnt][j]
                if self.df_seg.former_trans_mode[i] is 'bus':
                    for j in range(0, 4):
                        y_predict_probability[cnt][j] = tp[1][j] * y_predict_probability[cnt][j]
                if self.df_seg.former_trans_mode[i] is 'car':
                    for j in range(0, 4):
                        y_predict_probability[cnt][j] = tp[2][j] * y_predict_probability[cnt][j]
                if self.df_seg.former_trans_mode[i] is 'walk':
                    for j in range(0, 4):
                        y_predict_probability[cnt][j] = tp[3][j] * y_predict_probability[cnt][j]
            cnt += 1
        # 最大值的坐标
        self.y_predict = y_predict_probability.argmax(axis=1)

    def get_accuracy_score(self):
        """
        计算模型的准确率
        :return: model's accuracy score
        """

        # 在测试集上预测交通方式
        return accuracy_score(self.y_test, self.y_predict)

    def print_info_on_confusion_matrix(self):
        """
        - 输出模型的混淆矩阵
        - 计算 true positive, false positive, true negative, false negative
            true positive : 实际是A，判断结果是A
            false positive: 实际是B，判断结果是A
            true negative : 实际是B，判断结果是B
            false negative: 实际是A，判断结果是B
        - 计算精确率，召回率，真假类率，FR指标
        :return:
        """
        confusion = confusion_matrix(self.y_test, self.y_predict)

        print(confusion)

        # 计算每一种交通工具的混淆矩阵
        Total_instances = np.sum(confusion)

        bike_actuals = np.sum(confusion[0])
        bus_actuals = np.sum(confusion[1])
        car_actuals = np.sum(confusion[2])
        walk_actuals = np.sum(confusion[3])

        tp_bike = confusion[0, 0]
        tp_bus = confusion[1, 1]
        tp_car = confusion[2, 2]
        tp_walk = confusion[3, 3]

        fn_bike = bike_actuals - tp_bike
        fn_bus = bus_actuals - tp_bus
        fn_car = car_actuals - tp_car
        fn_walk = walk_actuals - tp_walk

        bike_predict = np.sum(confusion, axis=0)[0]
        bus_predict = np.sum(confusion, axis=0)[1]
        car_predict = np.sum(confusion, axis=0)[2]
        walk_predict = np.sum(confusion, axis=0)[3]

        fp_bike = bike_predict - tp_bike
        fp_bus = bus_predict - tp_bus
        fp_car = car_predict - tp_car
        fp_walk = walk_predict - tp_walk

        tn_bike = Total_instances - tp_bike - fn_bike - fp_bike
        tn_bus = Total_instances - tp_bus - fn_bus - fp_bus
        tn_car = Total_instances - tp_car - fn_car - fp_car
        tn_walk = Total_instances - tp_walk - fn_walk - fp_walk

        # 输出每一个交通工具的混淆矩阵
        print('--------------------------------------------------------')
        print('Total instances in confusion matrix: ', Total_instances)
        print('--------------------------------------------------------')
        print('Bike_predict: ', bike_predict)
        print('Bike_actuals: ', bike_actuals)
        print('')
        print('True Positive : ', tp_bike)
        print('True Negative : ', tn_bike)
        print('False Positive: ', fp_bike)
        print('False Negative: ', fn_bike)
        print('')
        print('--------------------------------------------------------')
        print('Bus_predict: ', bus_predict)
        print('Bus_actuals: ', bus_actuals)
        print('')
        print('True Positive : ', tp_bus)
        print('True Negative : ', tn_bus)
        print('False Positive: ', fp_bus)
        print('False Negative: ', fn_bus)
        print('')
        print('--------------------------------------------------------')
        print('Car_predict: ', car_predict)
        print('Car_actuals: ', car_actuals)
        print('')
        print('True Positive : ', tp_car)
        print('True Negative : ', tn_car)
        print('False Positive: ', fp_car)
        print('False Negative: ', fn_car)
        print('')
        print('--------------------------------------------------------')
        print('Walk_predict: ', walk_predict)
        print('Walk_actuals: ', walk_actuals)
        print('')
        print('True Positive : ', tp_walk)
        print('True Negative : ', tn_walk)
        print('False Positive: ', fp_walk)
        print('False Negative: ', fn_walk)
        print('')
        print('--------------------------------------------------------')

        """
        精确率：正确被判断的 / 实际被判断的
        召回率：正确被判断的 / 应该被判断的
        特异性指标：正确被判断到不是的 / 应该被判断到不是的
        F1： 综合精确率和召回率
        """
        accuracy_model = (tp_bike + tp_bus + tp_car + tp_walk) / float(Total_instances)

        bike_precision = tp_bike / float(bike_predict)
        bus_precision = tp_bike / float(bus_predict)
        car_precision = tp_bus / float(car_predict)
        walk_precision = tp_walk / float(walk_predict)

        bike_recall = tp_bike / float(bike_actuals)
        bus_recall = tp_bus / float(bus_actuals)
        car_recall = tp_car / float(car_actuals)
        walk_recall = tp_walk / float(walk_actuals)

        bike_specificity = tn_bike / float(tn_bike + fp_bike)
        bus_specificity = tn_bus / float(tn_bus + fp_bus)
        car_specificity = tn_car / float(tn_car + fp_car)
        walk_specificity = tn_walk / float(tn_walk + fp_walk)

        bike_falsepositive_rate = fp_bike / float(tn_bike + fp_bike)
        bus_falsepositive_rate = fp_bus / float(tn_bus + fp_bus)
        car_falsepositive_rate = fp_car / float(tn_car + fp_car)
        walk_falsepositive_rate = fp_walk / float(tn_walk + fp_walk)

        print('----------------------------------------------------------')
        print('Model accuracy:\t\t{0:.2f}'.format(accuracy_model))
        print('Model classification error:\t{0:.2f}'.format(1 - accuracy_model))
        print('----------------------------------------------------------')
        print('Bike recall (TP rate):\t{0:.2f}'.format(bike_recall))
        print('Bike FP Rate:\t\t{0:.2f}'.format(bike_falsepositive_rate))
        print('Bike precision:\t\t{0:.2f}'.format(bike_precision))
        print('Bike specif.\t\t{0:.2f}'.format(bike_specificity))
        print('----------------------------------------------------------')
        print('Bus recall (TP rate):\t{0:.2f}'.format(bus_recall))
        print('Bus FP Rate:\t\t{0:.2f}'.format(bus_falsepositive_rate))
        print('Bus precision:\t\t{0:.2f}'.format(bus_precision))
        print('Bus specif.:\t\t{0:.2f}'.format(bus_specificity))
        print('----------------------------------------------------------')
        print('Car recall (TP rate):\t{0:.2f}'.format(car_recall))
        print('Car FP Rate:\t\t{0:.2f}'.format(car_falsepositive_rate))
        print('Car precision:\t\t{0:.2f}'.format(car_precision))
        print('Car specif.:\t\t{0:.2f}'.format(car_specificity))
        print('----------------------------------------------------------')
        print('Walk recall (TP rate):\t{0:.2f}'.format(walk_recall))
        print('Walk FP Rate:\t\t{0:.2f}'.format(walk_falsepositive_rate))
        print('Walk precision:\t\t{0:.2f}'.format(walk_precision))
        print('Walk specif.:\t\t{0:.2f}'.format(walk_specificity))


if __name__ == '__main__':
    ip = ImproveModel()
    ip.get_probability()
    acc_score = ip.get_accuracy_score()
    print('Accuracy Score: ', acc_score)

    ip.print_info_on_confusion_matrix()

