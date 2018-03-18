# -*- coding: utf-8 -*-
import numpy as np

# x 값과 y값
x=[2, 4, 6, 8]
y=[81, 93, 91, 97]

# x와 y의 평균값
mx = np.mean(x)
my = np.mean(y)
# print("x의 평균값:", mx)
# print("y의 평균값:", my)

# 기울기 공식의 분모
# 기울기 분모: x의 각 원소에 x평균을 빼고 제곱 하여 합한다
divisor_bunMo_a = sum([(i - mx)**2 for i in x])

# 기울기 공식의 분자
# 배열x의 각 원소에 x평균을 빼고,
# 배열y의 각 원소에 y평균을 빼고, 이 둘을 곱한다.  이 값들을 다 더한다
def getBunMo_a(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d

numerator_dividend = getBunMo_a(x, mx, y, my)

print("분모:", divisor_bunMo_a)  # denominator 분모
print("분자:", numerator_dividend)

# 기울기와 y 절편 구하기
a = numerator_dividend / divisor_bunMo_a
b = my - (mx * a)

# 출력으로 확인
print("기울기 a =", a)
print("y 절편 b =", b)
