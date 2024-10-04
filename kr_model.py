from keras.models import Sequential
from keras.layers import Dense, Input
import pandas as pd
df = pd.read_csv('py4ai-score.csv')
for i in range(1,11):
  df.loc[:,f'S{i}'] = df.loc[:,f'S{i}'].fillna(0)

df['BONUS'] = df['BONUS'].fillna(0)
df['REG-MC4AI'] = df['REG-MC4AI'].fillna('N')
lb1 = {'N': 0, 'Y':1}
df['idREG']=df['REG-MC4AI'].map(lb1)
df['S-AVG']=df.loc[:,['S1','S2','S3','S4','S5','S7','S8','S9']].mean(axis=1)
xx = df[['S-AVG','GPA']].to_numpy()
yy = df['idREG'].to_numpy()
kr_model = Sequential()
kr_model.add(Input(shape=(xx.shape[1],))) 
kr_model.add(Dense(1, activation='sigmoid'))
kr_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = kr_model.fit(xx, yy, epochs=3000, verbose=0)
kr_model.save('model.keras')
