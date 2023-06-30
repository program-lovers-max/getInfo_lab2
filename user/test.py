import re

str = '计算机学院楼\n'
pattern = input()
if re.match(pattern,str):
    print('成功！')