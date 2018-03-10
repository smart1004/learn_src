# Day_01_01_python.py

# ctrl + shift + f10 : run
# ctrl + / : 주석
# shift + tab : 들여쓰기, 내어쓰기

a, b = 7, 3
print(a, b)

a, b = b, a     # 여러 변수 한번에 치환이 가능
print(a, b)
print('-'*50)

print('hello')
print("hello")
print('-'*50)

print(12, 3.14, 'hi', True)
print(type(12), type(3.14), type('hi'), type(True))
print('-'*50)


# 산술연산자
a, b = 9, 4
print(a+b)
print(a-b)
print(a*b)
print(a/b)      # 실수 나눗셈
print(a**b)     # 지수
print(a//b)     # 정수 나눗셈(몫)
print(a%b)      # 나머지

print('hello'+'python')
print('hello'* 3)
print('-'*50)

#  관계연산자
print(a > b)
print(a >= b)
print(a < b)
print(a <= b)
print(a == b)
print(a != b)

age = 15
print(10 <= age <= 19)      # 범위연산자, python 유일

print('-'*50)

# 논리연산자 : and, or, not
print(True and True)
print(True and False)
print(False and True)
print(False and False)
print('-'*50)



# a = input()     # 'str'로 입력 됨
# print(a)
# print(type(a))
#
# a = int(a)      # 파이썬은 데이터타입이 밖으로 노출되지 않는다. 'int'는 함수다
# print(type(a))

a = 23

if a % 2 ==1:
    print('odd')
else:
    print('even')

if a < 0:
    print('neg')
elif a > 0:
    print('pos')
else:
    print('zero')

print('-'*50)

# 0 1 2 3 4     0, 4, 1

for i in range(0, 5, 1):
    print(i, end=' ')
print()

for i in range(0, 5):
    print(i, end=' ')
print()

for i in range(5):
    print(i, end=' ')
print()

# 문제
# 0~4 사이의 숫자를 거꾸로 출력하세요.
for i in range(4, -1, -1):
    print(i, end=' ')
print()

for i in reversed(range(5)):        #파이썬 다운 코드
    print(i, end=' ')
print()
print('-'*50)

# shift + enter : 다음 줄

def f_1(t):
    print('f_1', t)

k = f_1(12)
# 파이썬은 기본 반환값 'None'이 있음. 없다가 아닌 없는 것을 가리키는 'False' 타입의 값
print(k)

def f_2(t1, t2):
    return t1 + t2


print(f_2(3,5))         # positional argument
# 함수를 호출하는 두 번째 방법
print(f_2(t1=3, t2=5))  # keyword argument
print(f_2(t2=3, t1=5))  # 이름을 부를 수 있으니 순서가 상관 없음
print(f_2(3, t2=5))     #
# print(f_2(t2=3, 5))  # error, positional은 keyword 앞에만 쓸 수 있음

# collection : list, tuple, dictionary
#              []      ()      {}
a = [1, 3, 5]
print(a)
print(a[0], a[1], a[2])     # index를 써서 접근을 하는 것

a.append(9)
print(len(a))       # collection과 같이 여러개가 있을 때는 개수를 추적할때

for i in range(len(a)):     #그래서 len을 써서 이렇게 하면 코드가 범용적임
    print(i, a[i])
print('-'*50)

for i in a:     # 무조건 전체를 다 써야함. 위의 것은 섞어서 쓸 수 있음
    print(i)
print('-'*50)
# 문제
# 리스트를 거꾸로 출력해보세요.
for i in reversed(range(len(a))):
    print(i, a[i])
print('-'*50)

for i in reversed(a):     # 무조건 전체를 다 써야함. 위의 것은 섞어서 쓸 수 있음
    print(i)
print('-'*50)
print(a)

# 문제
# 리스트를 거꾸로 뒤집어보세요.
a[0], a[3] = a[3], a[0]
a[1], a[2] = a[2], a[1]
print(a)

a[0], a[len(a)-1] = a[len(a)-1], a[0]
a[1], a[len(a)-2] = a[len(a)-2], a[1]
print(a)

for i in range(len(a)//2):
    a[i], a[len(a) - i-1] = a[len(a) - i-1], a[i]
print(a)
print('-'*50)

a[0] = 77
print(a)        # [77, 5, 3, 1] 값을 대입
print('-'*50)

# 듀플 : 리스트 상수 버전, (데이터 변경을 할 수 없다) 따라서 파이썬 내부적으로 쓴다. 우리는 안써도 됨.
# a = (1, 3, 5)
print(a)
print(a[0], a[1], a[2])     # tuple도 index를 통해서 접근 가능

# a[0] = 11 # TypeError: 'tuple' object does not support item assignment
# a.append(99)    # AttributeError: 'tuple' object has no attribute 'append'

t1 = (3, 5)
print(type(t1), t1)

t2 = 3, 5       # 괄호를 안써도 tuple, 치환하는 개수가 다르면 tuple로 저장
print(type(t2), t2)

t3, t4 = 3, 5
print(t3, t4)

t5 = t3, t4             # packing(데이터를 묶어줌)
print(t5)               # t5는 tuple로 묶였다.



def f_3(t1, t2):
    return t1 + t2, t1 * t2


t = f_3(3, 5)       # return은 tuple로만 받을 수 있음
print(t)
print(list(t))      # 바꾸려면 형변환 필요

t6, t7 = f_3(3, 5)
print(t6, t7)


def f_4(t1, t2):
    return [t1 + t2, t1 * t2]       # 괄호에 따라 형이 달라지나?? -> yes []면 list, ()면 tuple

k1 = f_4(3, 5)
print(k1)
print(type(k1))

k2, k3 = f_4(3, 5)
print(k2, k3)
print(type(k2))

# dictionary

# 영어 사전 : 영어 단어를 찾으면 한글 설명 나옴
# 영어 단어 : key
# 한글 설명 : value

#      key    value   key   value       : dictionary는 순서가 없음.
# d = {'name': 'hoon', 'age': 20}
d = dict(name='hoon', age=20)       # keywoard argument를 사용한다. dict 형변환을 함. key값인 name, age는 str 파입으로 변환됨

d['addr'] = 'suji'      # insert
d['addr'] = 'seoul'     # update

print(d)
print(d['name'], d['addr'])


































