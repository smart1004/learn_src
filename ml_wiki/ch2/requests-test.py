# 데이터 가져오기
import requests
r = requests.get("http://api.aoikujira.com/time/get.php")
# 텍스트 형식으로 데이터 추출하기
text = r.text
print(text)
# 바이너리 형식으로 데이터 추출하기
bin = r.content
print(bin)