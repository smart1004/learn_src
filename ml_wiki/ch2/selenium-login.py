from selenium import webdriver
USER = "<아이디>"
PASS = "<비밀번호>"
# PhantomJS 드라이버 추출하기 --- (※1)
browser = webdriver.PhantomJS()
browser.implicitly_wait(3)
# 로그인 페이지에 접근하기 --- (※2)
url_login = "https://nid.naver.com/nidlogin.login"
browser.get(url_login)
print("로그인 페이지에 접근합니다.")
# 텍스트 박스에 아이디와 비밀번호 입력하기 --- (※3)
e = browser.find_element_by_id("id")
e.clear()
e.send_keys(USER)
e = browser.find_element_by_id("pw")
e.clear()
e.send_keys(PASS)
# 입력 양식 전송해서 로그인하기 --- (※4)
form = browser.find_element_by_css_selector("input.btn_global[type=submit]")
form.submit()
print("로그인 버튼을 클릭합니다.")
# 쇼핑 페이지의 데이터 가져오기 --- (※5)
browser.get("https://order.pay.naver.com/home?tabMenu=SHOPPING")
# 쇼핑 목록 출력하기 --- (※6)
products = browser.find_elements_by_css_selector(".p_info span")
print(products)
for product in products:
    print("-", product.text)