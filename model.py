import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
sns.set_theme(style="whitegrid")

data=pd.read_excel('ENB2012_data.xlsx')

#print(data)
'''plt.figure(figsize=(20,10))
sns.boxplot(data=data, orient="v", palette="Set3")
sns.pairplot(data)
plt.savefig('new.jpg')
plt.show()'''
x=data.drop(['Y1','Y2'],axis=1)
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
X_scale=scale.fit_transform(x)
#print(X_scale)
X=pd.DataFrame(X_scale)
Y=data[['Y1','Y2']]
scale_data= pd.concat([X, Y], axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.15, random_state=42)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                          max_depth=30,
                                                          random_state=0))

model.fit(x_train, y_train)

y_multirf = model.predict(x_test)

pickle.dump(model,open('model.pkl','wb'))


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

n_scores = np.absolute(n_scores)

#print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


