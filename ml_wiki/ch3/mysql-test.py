# 라이브러리 읽어 들이기 --- (※1)
import MySQLdb
# MySQL 연결하기 --- (※2)
conn = MySQLdb.connect(
    user='root',
    passwd='test-password',
    host='localhost',
    db='test')
# 커서 추출하기 --- (※3)
cur = conn.cursor()
# 테이블 생성하기 --- (※4)
cur.execute('DROP TABLE items')
cur.execute('''
    CREATE TABLE items (
        item_id INTEGER PRIMARY KEY AUTO_INCREMENT,
        name TEXT,
        price INTEGER
    )
    ''')
# 데이터 추가하기 --- (※5)
data = [('Banana', 300),('Mango', 640), ('Kiwi', 280)]
for i in data:
    cur.execute("INSERT INTO items(name,price) VALUES(%s,%s)", i)
# 데이터 추출하기 --- (※6)
cur.execute("SELECT * FROM items")
for row in cur.fetchall():
    print(row)