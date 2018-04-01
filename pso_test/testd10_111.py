# 

'''

>>> param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
>>> clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
GridSearchCV(cv=None,
             estimator=LogisticRegression(C=1.0, intercept_scaling=1,   
               dual=False, fit_intercept=True, penalty='l2', tol=0.0001),
             param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})


TUTORIAL ON LOGISTIC REGRESSION AND OPTIMIZATION IN PYTHON
APRIL 30, 2015 TYLER	LEAVE A COMMENT
Go straight to the code

This post goes into some depth on how logistic regression works and 
how we can use gradient descent to learn our parameters. 
It also touches on how to use some more advanced optimization techniques in Python.


http://nbviewer.jupyter.org/github/tfolkman/learningwithdata/blob/master/Logistic%20Gradient%20Descent.ipynb

'''