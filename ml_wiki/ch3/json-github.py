import urllib.request as req
import os.path, random
import json
# JSON 데이터 내려받기 --- (※1)
url = "https://api.github.com/repositories"
savename = "repo.json"
if not os.path.exists(url):
    req.urlretrieve(url, savename)
# JSON 파일 분석하기 --- (※2)
items = json.load(open(savename, "r", encoding="utf-8"))
# 또는
# s = open(savename, "r", encoding="utf-8").read()
# items = json.loads(s)
# 출력하기 --- (※3)
for item in items:
    print(item["name"] + " - " + item["owner"]["login"])