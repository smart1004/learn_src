# Day_03_04_seaborn.py
import seaborn as sns
import matplotlib.pyplot as plt

def plot_1():
    print(sns.get_dataset_names())
    # BeautifulSoup은 파이썬에서 중요한 것. "html.parser"
    # ['anscombe', 'attention', 'brain_networks', 'car_crashes', 'dots',
    # 'exercise', 'flights', 'fmri', 'gammas', 'iris', 'planets', 'tips', 'titanic']


    iris = sns.load_dataset('iris')
    print(type(iris))
    print(iris)

    sns.swarmplot(x='species', y='petal_length', data=iris)
    plt.show()

def plot_2():
    # 선실 별로 생존자 데이터
    titanic = sns.load_dataset('titanic')
    sns.factorplot('class', 'survived', 'sex',
                   data=titanic, kind='bar',
                    palette='muted', legend=False)
    plt.show()

# seaborn: statistical data visualization
def plot_3():
    df = sns.load_dataset('anscombe')
    sns.lmplot(x='x', y='y', col='dataset',
                hue='dataset', data=df,
                col_wrap=2)
    plt.show()

plot_3()











