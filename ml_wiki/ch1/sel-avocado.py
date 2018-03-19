from bs4 import BeautifulSoup 
fp = open("fruits-vegetables.html", encoding="utf-8")
soup = BeautifulSoup(fp, "html.parser")
# CSS 선택자로 추출하기
print(soup.select_one("li:nth-of-type(8)").string)  #(※1)
print(soup.select_one("#ve-list > li:nth-of-type(4)").string)  #(※2)
print(soup.select("#ve-list > li[data-lo='us']")[1].string)  #(※3)
print(soup.select("#ve-list > li.black")[1].string)  #(※4)
# find 메서드로 추출하기 ---- (※5)
cond = {"data-lo":"us", "class":"black"}
print(soup.find("li", cond).string)
# find 메서드를 연속적으로 사용하기 --- (※6)
print(soup.find(id="ve-list")
           .find("li", cond).string)