# 이미지 데이터 추출하기
import requests
r = requests.get("http://wikibook.co.kr/wikibook.png")
# 바이너리 형식으로 데이터 저장하기
with open("test.png", "wb") as f:
    f.write(r.content)
print("saved")