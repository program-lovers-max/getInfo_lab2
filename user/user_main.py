import glob
import os
import csv
teacher_path = '../CV_cabin_process'

def getInfo(info_vector):
    for teacher_info in glob.glob(os.path.join(teacher_path, "*.txt")):
        teacher_name = os.path.basename(teacher_info).strip().split('.')[0]


if __name__ == '__main__':
    print('您想查找的信息点是:(教师姓名、所在组织、学科、学位、头衔职位)')
    print('请用一个五维向量标识您的查询，如想要获取（姓名、学科）信息，则输入’1 0 1 0 0‘')
    print('请输入您想查询的信息点:')
    info_vector = input().split()
    getInfo(info_vector)