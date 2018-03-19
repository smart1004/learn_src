# IP 확인 API로 접근해서 결과 출력하기
# 모듈 읽어 들이기 --- (※1)
import urllib.request
# 데이터 읽어 들이기 --- (※2)
url = "http://api.aoikujira.com/ip/ini"
res = urllib.request.urlopen(url)
data = res.read()
# 바이너리를 문자열로 변환하기 --- (※3)
text = data.decode("utf-8")
print(text)