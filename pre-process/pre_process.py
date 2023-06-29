import glob
import os

# 存储教师信息的目录
teacher_path = '../CV_cabin_origin'
# 遍历文件夹下的所有文本文件
if __name__ == '__main__':
    for teacher_info in glob.glob(os.path.join(teacher_path, "*.txt")):
        teacher_name = os.path.basename(teacher_info).strip().split('.')[0]
        flag = False
        with open(teacher_info, mode='r', encoding='utf-8') as teacher:
            url = teacher.readline()
            content = teacher.read()
            if not content.startswith(('个人',teacher_name)):
                flag = True
            content = teacher_name+content
        with open(teacher_info, mode='w', encoding='utf-8') as teacher:
            teacher.write(url  + content)
        if flag:
            os.remove(teacher_info)
