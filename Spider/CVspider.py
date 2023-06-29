import os
from bs4 import BeautifulSoup
import re
import requests
from lxml import etree
headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36 Edg/105.0.1343.27"
    }  #设置头部内容，可以反反爬虫
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
    result = html.xpath('//*[@id="vsb_content"]/div/table/tbody/tr/td/a')
    for a in result:
        name = a.attrib['title']
        url = a.attrib['href']
        if url!='#' and url!='':
            teacher_url[name] = url
    return teacher_url


def get_teacher_info(teacher_url):
    for name,url in teacher_url.items():
        if url.startswith("hhttp"):
            url = url.replace("hhttp", "http")
        response = requests.get(url,headers=headers)
        response.encoding = response.apparent_encoding
        html_content = response.text  # 生成可以xpath的html信息树
        soup = BeautifulSoup(html_content, 'html.parser')
        paragraphs = soup.find_all(['p', 'td', 'span','div'])
        filtered_content =  filter_strings([tag.get_text() for tag in paragraphs])
        longest_string = str(max(filtered_content, key=len))
        longest_string_tight = re.sub(r'[\s\da-zA-Z]', '', longest_string)

        with open(f'../CV_cabin_origin/{name}.txt', mode='w', encoding='utf-8') as info:
            info.write(longest_string_tight)
if __name__ == '__main__':
    if not os.path.exists('../CV_cabin_origin'):
        os.mkdir('../CV_cabin_origin')
    department_url = 'https://scs.bupt.edu.cn/szjs1/jsyl.htm'
    teacher_url = get_teacher_url(department_url)
    get_teacher_info(teacher_url)