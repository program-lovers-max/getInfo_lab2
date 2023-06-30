import os
import random
import time

from bs4 import BeautifulSoup
import re
import requests
from lxml import etree
# 设置随机agent列表，用于反爬
agent_list = [ "Mozilla/5.0 (Linux; U; Android 2.3.6; en-us; Nexus S Build/GRK39F) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
               "Avant Browser/1.2.789rel1 (http://www.avantbrowser.com)",
               "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/532.5 (KHTML, like Gecko) Chrome/4.0.249.0 Safari/532.5",
               "Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US) AppleWebKit/532.9 (KHTML, like Gecko) Chrome/5.0.310.0 Safari/532.9",
               "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/534.7 (KHTML, like Gecko) Chrome/7.0.514.0 Safari/534.7",
               "Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/534.14 (KHTML, like Gecko) Chrome/9.0.601.0 Safari/534.14",
               "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.14 (KHTML, like Gecko) Chrome/10.0.601.0 Safari/534.14",
               "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.20 (KHTML, like Gecko) Chrome/11.0.672.2 Safari/534.20",
               "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.27 (KHTML, like Gecko) Chrome/12.0.712.0 Safari/534.27",
               "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/13.0.782.24 Safari/535.1",
               "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/535.2 (KHTML, like Gecko) Chrome/15.0.874.120 Safari/535.2",
               "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.7 (KHTML, like Gecko) Chrome/16.0.912.36 Safari/535.7",
               "Mozilla/5.0 (Windows; U; Windows NT 6.0 x64; en-US; rv:1.9pre) Gecko/2008072421 Minefield/3.0.2pre",
               "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.10) Gecko/2009042316 Firefox/3.0.10" ]
headers = {
        "user-agent": random.choice(agent_list)
    }  #随机从列表中取user-agent
# proxies = {
#     "http": "http://127.0.0.1:7890",
#     "https": "http://127.0.0.1:7890",
# }
def count_chars(text):
    count_en = 0  # 英文字符计数
    count_num = 0  # 数字字符计数
    count_zh = 0  # 汉字计数

    for char in text:
        if char.isascii():  # 判断是否为字母
            count_en += 1
        elif char.isdigit():  # 判断是否为数字
            count_num += 1
        elif '\u4e00' <= char <= '\u9fff':  # 判断是否为汉字
            count_zh += 1

    return count_en, count_num, count_zh

def filter_strings(string_list):
    filtered_list = []
    for s in string_list:
        count_en, count_num, count_zh = count_chars(s)
        if count_en + count_num < count_zh:
            filtered_list.append(s)
    return filtered_list

def get_teacher_url(department_url):
    teacher_url  = {}
    response = requests.get(department_url,headers = headers)  #获得响应报文
    response.encoding = response.apparent_encoding
    html = etree.HTML(response.text,etree.HTMLParser())  #生成可以xpath的html信息树
    result = html.xpath('//*[@id="vsb_content"]/div/table/tbody/tr/td/a') #获取存储姓名和url标签列表
    for a in result:
        name = a.attrib['title'] #提取姓名
        url = a.attrib['href']  #提取url
        if url!='#' and url!='':
            teacher_url[name] = url  #删去不合法的项
    return teacher_url


def get_teacher_info(teacher_url):
    for name,url in teacher_url.items():
        if url.startswith("hhttp"):    #修正url
            url = url.replace("hhttp", "http")
        response = requests.get(url,headers=headers)
        response.encoding = response.apparent_encoding
        html_content = response.text  # 生成可以xpath的html信息树
        soup = BeautifulSoup(html_content, 'html.parser')
        paragraphs = soup.find_all(['p', 'td', 'span','div'])
        filtered_content =  filter_strings([tag.get_text() for tag in paragraphs])  #筛选对应标签内的文本
        longest_string = str(max(filtered_content, key=len)) #只保留最长文本（即个人信息文本）
        longest_string_tight = re.sub(r'[\s\da-zA-Z]', '', longest_string)  #删去字母、数字、空格和回车以便bert处理

        with open(f'../CV_cabin_origin/{name}.txt', mode='w', encoding='utf-8') as info:
            info.write(url+'\n')  #写入url
            info.write(longest_string_tight) #写入处理后的文本
        time.sleep(random.uniform(0.5, 1))
if __name__ == '__main__':
    if not os.path.exists('../CV_cabin_origin'):
        os.mkdir('../CV_cabin_origin')
    department_url = 'https://scs.bupt.edu.cn/szjs1/jsyl.htm'
    teacher_url = get_teacher_url(department_url)
    get_teacher_info(teacher_url)