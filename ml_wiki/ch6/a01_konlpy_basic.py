
from konlpy.tag import Twitter
twitter = Twitter()
malist = twitter.pos("아버지 가방에 들어가신다", norm=True, stem=True)
print(malist)
# [('아버지', 'Noun'), ('가방', 'Noun'), ('에', 'Josa'), ('들어가다', 'Verb')]
