# -*- coding: utf-8 -*-
"""
Module:     Modeling
Summary:    建立并训练模型
Author:     Yuhao Jiang
Created:    2017/12/13  Ver.0.1
Update:     2017/12/14  Ver.1.0
            2017/12/20  Ver.1.1 Add former_trans_mode
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

class ModelingProcessing(object):
    """
    基于GPS数据判断交通方式
    """

    def __init__(self, segment_featured_file_path):
        self.df_seg = pd.read_csv(segment_featured_file_path, index_col=0)
        self.rf = None
        self.X_test_final = None
        self.y_test = None

    def extract_trans_mode(self):
        """
        从df_seg提取交通方式
        :return:
        """
        self.df_seg['trans_mode'] = (self.df_seg['segment_ID']).str[24:]

    def reduce_low_sample_modes(self):
        """
        删除boat, run, airplane, train, subway等低频交通方式
        :return:
        """
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'boat']
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'run']
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'airplane']
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'train']
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'subway']
        self.df_seg = self.df_seg[self.df_seg.trans_mode != 'motorcycle']





    def model(self):
        """
        - 定义特征矩阵，预测结果向量化
        - 整合所有类型
        - 把字符串数据转化成数值
        - 分割出训练集，测试集
        - 训练随机森林分类器
        :return:
        """

        # 1.创建特征矩阵X
        # 包含所有计算数据
        X = self.df_seg.loc[:, 'segment_ID':'bearing_change_rate']

        # 2.创建结果矩阵，向量化
        y = self.df_seg['trans_mode']

        # 3.整合类型
        y[y == 'taxi'] = 'car_taxi'
        y[y == 'car'] = 'car_taxi'

        # 4.给四个类型向量化
        # ['bike', 'bus', 'car_taxi', 'walk']
        # [   0      1        2          3  ]
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(np.unique(y))
        y = label_encoder.transform(y)

        # 5.分割出训练集和测试集，25%测试集，75%训练集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        print('Training Set (X_train, y_train): ', X_train.shape, y_train.shape)
        print('Testing Set (X_test, y_test): ', X_test.shape, y_test.shape)

        # 6.修改测试集，训练集

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

        X_train_final = X_train[feature_cols]
        self.X_test_final = X_test[feature_cols]
        self.y_test = y_test

        # 7.创建随机森林分类器模型
        self.rf = RandomForestClassifier(criterion='entropy', max_features='sqrt', n_estimators=100, n_jobs=1)
        self.rf.fit(X_train_final, y_train)

    def get_accuracy_score(self):
        """
        计算模型的准确率
        :return: model's accuracy score
        """

        # 在测试集上预测交通方式
        y_predict_class = self.rf.predict(self.X_test_final)
        return accuracy_score(self.y_test, y_predict_class)

    def get_null_accuracy_score(self):
        """
        计算准确率：通过频率直接判断交通工具
        :return:
        """
        max_ocurr = max(np.bincount(self.y_test))
        sum_test = sum(self.y_test)

        return max_ocurr / float(sum_test)

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

        y_predict_class = self.rf.predict(self.X_test_final)
        confusion = confusion_matrix(self.y_test, y_predict_class)

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

    def pickle_model(self, pickle_path):
        """
        存储训练好的模型
        :param pickle_path:
        :return:
        """
        pickle.dump(self.rf, open(pickle_path, 'wb'))


if __name__ == '__main__':
    mp = ModelingProcessing('segment_featured_master.csv')

    mp.extract_trans_mode()
    mp.reduce_low_sample_modes()
    mp.model()

    acc_score = mp.get_accuracy_score()
    print('Accuracy Score: ', acc_score)

    null_acc_score = mp.get_null_accuracy_score()
    print('Null Accuracy Score: ', null_acc_score)

    mp.print_info_on_confusion_matrix()

    mp.pickle_model('transport_classifier.pkl')