# -*- coding: utf-8 -*-
"""
Module:     PLT2CSV
Summary:    把plt，txt文件统一转换成csv格式
Author:     Yuhao Jiang
Created:    2017/12/11  Ver.1.0
"""
import os
class ChangeFormat(object):

    def change(self, dirpath):
        dirpaths = os.listdir(dirpath)
        for dirname in dirpaths:
            filepath = os.path.join(dirpath, dirname, 'Trajectory')
            filenames = os.listdir(filepath)

            for filename in filenames:
                portion = os.path.splitext(filename)

                if portion[1] == '.plt' or portion[1] == '.txt':
                    newname = portion[0] + '.csv'
                    filename_path = os.path.join(filepath, filename)
                    newname_path = os.path.join(filepath, newname)
                    os.rename(filename_path, newname_path)


if __name__ == "__main__":
    file = ChangeFormat()
    dirpath = 'D:\Zhejiang University\Graduate Project\Data\Geolife Trajectories 1.3\Data'
    file.change(dirpath)