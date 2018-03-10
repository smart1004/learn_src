# Day_02_01_numpy.py

import numpy as np

def numpy_1():
    # seed를 고정 시키면, 같은 난수가 나옴. seed 비고정시에는 계속 다른 값을 나옴
    np.random.seed(1)

    a = np.random.choice(range(100),12)
    print(a)
    # choice는 일반적인 random은 아니다.
    # 뒤에 형태를 정확히 [] 내에 써야 한다.
    a = np.random.choice(range(100), [3, 4])
    print(a)

    a = np.random.choice([1, 3, 9], 5)
    print(a)
    # 딥러닝 시 임의의 값으로 초기화 시킬때 쓰는 함수 rand
    a = np.random.rand(5)
    print(a)
    # 범위가 지정 가능
    a = np.random.randint(5, 10, [2,3])
    print(a)
    print('-' * 50)
    # 문제
    # choice 함수를 사용하지 말고
    # [1, 3, 9]에서 5개의 값을 랜덤하게 뽑으세요.
    # 주어진 선택지 내에서 원하는 만큼 자유롭게 뽑을 수 있으면 해결
    index = np.random.randint(0, 3, 5)
    print(index)
    for i in index:
        print([1, 3, 9][i], end=' ')
    print()

def numpy_2():
    a = np.arange(12).reshape(3, -1)
    print(a)

    # # 파이썬 합계와 ndarray 섞어쓰기 금지
    # print(sum(a))   # sum  함수는 파이썬 기본 제공 함수 임
    # print(sum([1, 3, 9]))

    # ndarray 의 행, 열의 합을 각각 구하는 것이 중요함
    print(np.sum(a))
    print(a.sum())

    print(np.sum(a, axis=0))        # 수직(열) 합
    print(np.sum(a, axis=1))        # 수평(행) 합
    # 0, 1이 행,열로 보면 바뀐 것 같은데 열을 기준으로 생각하면 이해할 수 있음

    # 문제
    # a에서 가장 큰 숫자를 찾으세요. 3가지
    np.random.seed(1)
    a = np.random.choice(range(100),[3,4])
    print(a)
    # 하나씩 비교를 하는 방법 또는 새로운 함수
    print(a.max())
    print(a.max(axis=0))
    print(a.max(axis=1))


a = np.arange(12).reshape(3, 4)
print(a)

print(a[0])
print(a[-1])
print(a[:2])    # 앞쪽의 2개만 꺼내올때.

a[0] = -1       # -1이 다 들어감 [-1 -1 -1 -1]. broadcasting으로 볼 수 있음
print(a)

b = [[1, 2], [3, 4]]
print(b)
print(b[0])
b[0] = -1       # [-1, [3, 4]] , list를 밀어 내고 int가 들어감.
print(b)
print('-' * 50)

a[:2] = 99      # slicing을 통해서 부분만도 바꿀 수 있음
print(a)
a[::2] = 7      # slicing을 통해서 부분만도 바꿀 수 있음
print(a)
print('-' * 50)

print(a[0] + a[2])                  # 백터 연산
print([1, 2, 3] + [1, 2, 3])        # 리스트 확장
print('-' * 50)


a = np.arange(12).reshape(3, 4)
b = np.arange(12).reshape(3, 4)
print(a)
print(b)
# 두 array를 연결하고 싶을 때
print(np.vstack([a,b]))
print(np.hstack([a,b]))

print(a[::-1])          # 행을 역순함
print(a[::-1][::-1])    # 행을 역, 역하여 초기값으로 회귀

print(a[0])             # [0 1 2 3]
print(a[0][0], a[-1][-1])       # 연산을 두번 하여 찾음
print(a[0, 0], a[-1, -1])       # fancy indexing

# 문제
# fancy 인덱싱을 사용해서 거꾸로 출력해보세요.
# [0,0] -> [3, 3] [0, 1] ->[2,2] [0,2] -> [2, 1]
print(a[::-1, ::-1])    # 행, 열에 모두 적용 가능 할 수 있다.
print('-' * 50)


# 문제
# 반복문을 사용해서 거꾸로 출력해보세요.
for i in a[::-1]:
    for j in i[::-1]:
        print(j, end=' ')

    print()

for i in reversed(a):
    for j in reversed(i):
        print(j, end=' ')
    print()
print('-' * 50)


print(a)
print(np.transpose(a))      # 행, 열을 바꾼다

# print(np.dot(a, a))       # error. 대상의 행열을 맞춰야 함
print(np.dot(a, a.transpose()))
print(np.dot(a.T, a))       # a.T 긴것을 .T로 단축어 사용

# 문제
# 2차원 리스트를 반복문을 사용하여
# transpose 형태로 출력해주세요.

# a = [[1,2], [3,4]]
# print(a[1,1])
# a = np.arange(4).reshape(2, 2)
# print(type(a))
# a = list(a)
# print(a)
# print(type(a))

for i in range(a.shape[1]):
    for j in range(a.shape[0]):
        print(j, i, end=', ')
    print()


for i in range(a.shape[1]):
    for j in range(a.shape[0]):
        print(a[j, i], end=', ')
    print()


for i in range(a.shape[1]):
    print(a[:, i])
print('-' * 50)


# 문제
# 테두리가 1로 채워지고, 속이 0으로 채워진
# 4행 5열 배열을 만드세요.
a1 = np.zeros([4,5], dtype=np.int32)
a1[0], a1[-1] = 1, 1
a1[:, 0], a1[:, -1] = 1, 1      # 0번째 열의 모든 행에 1을 넣겠다 and  마지막 열의 모든 행에 1을 넣겠다

print(a1)
print(a1[ [2, 0, 1] ])          # 인덱스 배열, 2, 0, 1번째를 순서대로 들고 오겠다는 의미

a1[ [0, -1] ] = 1
a1[:, [0, -1] ] = 1
print('-' * 50)

a2 = np.ones([4,5], dtype=np.int32)
a2[1:-1, 1:-1] = 0      # slicing 문법을 사용 하여 범위를 구분함, slicing에 한하여 -1은 제외한다 (이해가 잘 안됨)
print(a2)
print('-' * 50)

print(np.eye(5, 5, dtype=np.int32))
# 문제
# 변수 a를 만들어서
# eye 함수와 똑같이 채워주세요.
#print(np.zeros([5, 5], dtype=np.int32))
a3 = np.zeros([5, 5], dtype=np.int32)
for i in range(a3.shape[1]):
    for j in range(a3.shape[0]):
        if i == j:
            a3[i, j] = 1
        else:
            a3[i,j]
print(a3)
print('-' * 50)

#강사코드
a3 = np.zeros([5, 5], dtype=np.int32)
a3[ [0, 1, 2, 3, 4], [0, 1, 2, 3, 4] ] = 1      # 인덱스 배열,
print(a3)
print('-' * 50)

a3[range(5), range(5)]  = 1
print(a3)
print('-' * 50)

x = np.array([3, 1, 2])
print(np.sort(x))       # 정렬을 하지만 데이터가 바뀌는 것은 아니다. sort는 잘 안씀
print(x)
print('-' * 50)

b = np.argsort(x)       # 데이터의 크기에 따른 인덱스를 뽑아줌.
print(b)
print(x[b])

# 문제
# 어떤 값이 나올지 예측해보세요.
x = np.array( [4, 3, 1, 5, 2])
print(np.argsort(x))  # 2 4 1 0 3









