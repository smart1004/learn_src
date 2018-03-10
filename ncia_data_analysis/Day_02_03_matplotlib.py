# Day_02_03_matplotlib.py
import numpy as np
import matplotlib.pyplot as plt

def random_walker_1():
    np.random.seed(42)
    walks, heights, pos = [], [], 0
    for _ in range(100):
        state = np.random.randint(-1, 2)
        # print(state)
        walks.append(state)

        pos += state
        heights.append(pos)

    plt.subplot(1, 2, 1)
    # plt.plot(walks)
    plt.plot(walks,'ro')

    plt.subplot(1, 2, 2)
    plt.plot(heights)
    plt.show()

# 문제
# 반복문 없이 같은 결과를 만들어 주세요.
# cumsum() 사용
def random_walker_2():
    np.random.seed(42)
    walks = np.random.randint(-1, 2, 100)
    heights = np.cumsum(walks)

    plt.subplot(1, 2, 1)
    plt.plot(walks,'ro')

    plt.subplot(1, 2, 2)
    plt.plot(heights)
    plt.show()

def style_1():
    print(plt.style.available)
    print(len(plt.style.available))     # 25 or 23개
    # ['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale',
    # 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', '
    # seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper',
    # 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white',
    # 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', '_classic_test']

    x = np.linspace(0, 10)
    print(x.shape)

    # with context manager : 아래있는 것은 with 문에 영향을 받는다라는 의미
    with plt.style.context('ggplot'):
        plt.plot(x, np.sin(x) + 0.5 * x + np.random.randn(len(x)))
        plt.plot(x, np.sin(x) + 1.0 * x + np.random.randn(len(x)))
        plt.plot(x, np.sin(x) + 1.5 * x + np.random.randn(len(x)))

    plt.show()

# 문제
# 제공하는 모든 스타일을 하나의 피겨에 그려주세요.
# 서브플랏을 25개 만들어 주세요.
# for i in x:
#     with plt.style.context('ggplot'):
#         plt.subplot(1, len(x), i)
#         plt.plot(x, np.sin(x) + 0.5*i * x + np.random.randn(len(x)))
#         # plt.plot(x, np.sin(x) + 1.0 * x + np.random.randn(len(x)))
#         # plt.plot(x, np.sin(x) + 1.5 * x + np.random.randn(len(x)))
#
#     plt.show()
#
# 강사코드
x = np.linspace(0, 10)
#
# for style in enumerate(plt.style.available):
#     print(style)

plt.figure(figsize=[20,15])     # size 조절, 단위 inch
for i, style in enumerate(plt.style.available):
    print(i, style)

    # plt.figure(i+1)            # 갯수 대로 plot

    with plt.style.context(style):
        plt.subplot(5, 5, i + 1)
        plt.plot(x, np.sin(x) + 0.5 * x + np.random.randn(len(x)))
        plt.plot(x, np.sin(x) + 1.0 * x + np.random.randn(len(x)))
        plt.plot(x, np.sin(x) + 1.5 * x + np.random.randn(len(x)))


plt.tight_layout()                  # 여백을 없애는 함수
# plt.show()                        # show를 하고
plt.savefig('Data/style.png')       # 현재 그림을 저장




















