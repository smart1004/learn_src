# 파일 이름과 데이터
filename = "a.bin"
data = 100
# 쓰기
with open(filename, "wb") as f:
    f.write(bytearray([data]))