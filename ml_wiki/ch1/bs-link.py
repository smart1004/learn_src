from bs4 import BeautifulSoup 
html = """
<html><body>
  <ul>
    <li><a href="http://www.naver.com">naver</a></li>
    <li><a href="http://www.daum.net">daum</a></li>
  </ul>
</body></html>
"""
# HTML 분석하기 --- (※1)
soup = BeautifulSoup(html, 'html.parser')
# find_all() 메서드로 추출하기 --- (※2)
links = soup.find_all("a")
# 링크 목록 출력하기 --- (※3)
for a in links:
    href = a.attrs['href']
    text = a.string
    print(text, ">", href)