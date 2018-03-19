import csv, codecs
with codecs.open("test.csv", "w", "euc_kr ") as fp:
    writer = csv.writer(fp, delimiter=",", quotechar='"')
    writer.writerow(["ID", "이름", "가격"])
    writer.writerow(["1000", "SD 카드 ", 30000])
    writer.writerow(["1001", "키보드", 21000])
    writer.writerow(["1002", "마우스", 15000])