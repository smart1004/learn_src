import requests
import json
# API 키를 지정합니다. 자신의 키로 변경해서 사용해주세요. --- (※1)
apikey = "474d59dd890c4108f62f192e0c6fce01"
# 날씨를 확인할 도시 지정하기 --- (※2)
cities = ["Seoul,KR", "Tokyo,JP", "New York,US"]
# API 지정 --- (※3)
api = "http://api.openweathermap.org/data/2.5/weather?q={city}&APPID={key}"
# 켈빈 온도를 섭씨 온도로 변환하는 함수 --- (※4)
k2c = lambda k: k - 273.15
# 각 도시의 정보 추출하기 --- (※5)
for name in cities:
    # API의 URL 구성하기 --- (※6)
    url = api.format(city=name, key=apikey)
    # API에 요청을 보내 데이터 추출하기
    r = requests.get(url)
    # 결과를 JSON 형식으로 변환하기 --- (※7)
    data = json.loads(r.text)    
    # 결과 출력하기 --- (※8)
    print("+ 도시 =", data["name"])
    print("| 날씨 =", data["weather"][0]["description"])
    print("| 최저 기온 =", k2c(data["main"]["temp_min"]))
    print("| 최고 기온 =", k2c(data["main"]["temp_max"]))
    print("| 습도 =", data["main"]["humidity"])
    print("| 기압 =", data["main"]["pressure"])
    print("| 풍향 =", data["wind"]["deg"])
    print("| 풍속 =", data["wind"]["speed"])
    print("")