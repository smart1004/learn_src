# Day_03_03_mpld3.py

import pandas as pd
import matplotlib.pyplot as plt
import mpld3
# matplot이 안 이뻐서...

def sample_plot(df, subject):
    df.plot(kind='line', marker='p', color=['blue', 'red'],
            lw=3, ms=20, alpha=0.7)

    plt.title(subject)
    plt.text(s='blue line', x=1, y=2, color='blue')
    plt.text(s='red line', x=2.7,y=3, color='red')


c1 = [1, 2, 3, 4]
c2 = [1, 4, 2, 3]

df = pd.DataFrame({'c1' : c1, 'c2': c2})

# sample_plot(df, 'base')

# plt.xkcd()
# sample_plot(df, 'xkcd')
# plt.show()


# web 상에서 표기가 됨.
sample_plot(df, 'd3.js')        # js : javascript기반
mpld3.show()











