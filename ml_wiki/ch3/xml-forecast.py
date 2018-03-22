from bs4 import BeautifulSoup 
import urllib.request as req
import os.path
url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108"
savename = "forecast.xml"
if not os.path.exists(savename):
    req.urlretrieve(url, savename)
# BeautifulSoup로 분석하기 --- (※2)
xml = open(savename, "r", encoding="utf-8").read()
soup = BeautifulSoup(xml, 'html.parser')
# 각 지역 확인하기 --- (※3)
info = {}
for location in soup.find_all("location"):
    name = location.find('city').string
    weather = location.find('wf').string
    if not (weather in info):
        info[weather] = []
    info[weather].append(name)
# 각 지역의 날씨를 구분해서 출력하기
for weather in info.keys():
    print("+", weather)
    for name in info[weather]:
        print("| - ", name)