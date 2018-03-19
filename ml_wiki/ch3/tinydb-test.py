# TinyDB를 사용하기 위한 라이브러리 읽어 들이기
from tinydb import TinyDB, Query
# 데이터베이스 연결하기 --- (※1)
filepath = "test-tynydb.json"
db = TinyDB(filepath)
# 기존의 테이블이 있다면 제거하기 --- (※2)
db.purge_table('fruits')
# 테이블 생성/추출하기 --- (※3)
table = db.table('fruits')
# 테이블에 데이터 추가하기 --- (※4)
table.insert( {'name': 'Banana', 'price': 6000} )
table.insert( {'name': 'Orange', 'price': 12000} )
table.insert( {'name': 'Mango', 'price': 8400} )
# 모든 데이터를 추출해서 출력하기 --- (※5)
print(table.all())
# 특정 데이터 추출하기
# Orange 검색하기 --- (※6)
Item = Query()
res = table.search(Item.name == 'Orange')
print('Orange is ', res[0]['price'])
# 가격이 8000원 이상인 것 추출하기 --- (※7)
print("8000원 이상인 것:")
res = table.search(Item.price >= 800)
for it in res:
    print("-", it['name'])