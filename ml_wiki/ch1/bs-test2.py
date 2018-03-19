from bs4 import BeautifulSoup 
html = """
<html><body>
  <h1>스크레이핑이란?</h1>
  <p>웹 페이지를 분석하는 것</p>
  <p>원하는 부분을 추출하는 것</p>
</body></html>
"""
# HTML 분석하기 --- (※1)
soup = BeautifulSoup(html, 'html.parser')
# find() 메서드로 원하는 부분 추출하기 --- (※2)
title = soup.find(id="title")
body  = soup.find(id="body")
# 텍스트 부분 출력하기
print("#title=" + title.string)
print("#body="  + body.string)