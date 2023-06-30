import glob
import os
import csv
import re
from tkinter import Tk, Entry, Button, Label
import speech_recognition as sr

teacher_path = '../CV_cabin_process'
point_name = ['NAME','ORG','PRO','EDU','TITLE']
info_dic = {}
result_dic={}
def getInfo(point_dic):
    global info_dic
    valid_point = [key for key,value in point_dic.items() if value=='1']
    for teacher_info in glob.glob(os.path.join(teacher_path, "*.txt")):
        teacher_name = os.path.basename(teacher_info).strip().split('.')[0]
        with open(teacher_info, mode='r', encoding='utf-8') as teacher:
            url = teacher.readline().rstrip()
            info_dic[teacher_name] = [url]
            content = teacher.readline()
            while content != '':
                tmp_storage = ''
                end_flag = False
                line = content.split()
                for item in valid_point:
                    if content!= '' and item in line[1] and 'B-' in line[1]:
                        tmp_storage = line[0]
                        content = teacher.readline()
                        line = content.split()
                        while content !='' and ('I-' in line[1] or 'E-' in line[1]):
                            tmp_storage += line[0]
                            if 'E-' in line[1]:
                                end_flag = True
                                break
                            content = teacher.readline()
                            line = content.split()
                    if end_flag:
                       info_dic[teacher_name].append(tmp_storage)
                       break
                content = teacher.readline()
    for key in info_dic:
        info_dic[key] = list(dict.fromkeys(info_dic[key]))
    info_dic = {key: value for key, value in info_dic.items() if len(value) != 1}



def searchInfo(search_pattern):
    global result_dic
    for key,value in info_dic.items():
        result_dic[key] = [value[0]]
        for i in range(1,len(value)):
            if re.match(search_pattern,value[i]):
                result_dic[key].append(value[i])
    result_dic = {key: value for key, value in result_dic.items() if len(value) != 1}


def showInfo():
    header = ["教师姓名", "匹配内容", "URL"]  # 设置表头
    with open('./result.csv', 'w', encoding='utf-8', newline='') as result_csv:
        writer = csv.writer(result_csv)
        writer.writerow(header)
        for key,value in result_dic.items():
            name = key
            URL = value[0]
            content = value[1:]
            info = [name,content,URL]
            writer.writerow(info)
    result_csv.close()
    os.system('start ./result.csv')

def submit():
    entered_text = entry.get()
    print("用户对本次搜索结果的评分是:", entered_text)

if __name__ == '__main__':
    print('您想查找的信息点是:(教师姓名、所在组织、学科、学位、头衔职位)')
    print('请用一个五维向量标识您的查询，如想要获取（姓名、学科）信息，则输入’1 0 1 0 0‘')
    print('请输入您想查询的信息点:')
    point_vector = input().split()
    point_dic = {k:v for k,v in zip(point_name,point_vector)}
    getInfo(point_dic)
    print('请输入您关心的具体信息【可以使用正则表达式】')
    print('如您可以输入 ’计算机.*学院‘ 表达式搜索所有符合该正则表达式的信息点信息')
    print('请输入您想查询的信息:')
    r = sr.Recognizer()
    print("-------控制台文本输入请按0，语音输入请按1---------")
    choice = int(input())
    if choice:
        with sr.Microphone() as source:
            while True:
                print('请说出你想查询的词或句')
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                try:
                    sentence = r.recognize_sphinx(audio,language='zh-CN')
                    print(f'您的语音输入是{sentence}')
                    searchInfo(sentence)
                    showInfo()
                    break
                except Exception as e:
                    print('未能识别你的语音，请重试')
    else:
        print('请输入你想要查询的文本(可使用正则)')
        search_pattern = input()
        print(search_pattern)
        searchInfo(search_pattern)
        showInfo()
    # 创建主窗口
    root = Tk()

    # 设置窗口大小
    root.geometry("500x300")

    # 创建标签
    label = Label(root, text="请输入你对本次搜索结果的评分(0-10)")
    label.pack()

    # 创建输入框
    entry = Entry(root)
    entry.pack()

    # 创建按钮
    button = Button(root, text="提交", command=submit)
    button.pack()

    # 设置布局管理器
    label.pack(pady=40)  # 添加一些垂直间距
    entry.pack(pady=20)
    button.pack(pady=20)
    # 进入事件循环
    root.mainloop()