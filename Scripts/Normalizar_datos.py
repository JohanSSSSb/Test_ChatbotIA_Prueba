import pandas as pd



df = pd.read_csv('./Datos/dataset_limpio.csv', delimiter=',')


df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)

df = df.drop_duplicates()


df.to_csv('datase_limpio.csv', index=False)

print("Success.........'")