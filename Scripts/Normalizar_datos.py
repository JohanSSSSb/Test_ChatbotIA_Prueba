import pandas as pd



df = pd.read_csv('./Datos/dataset_limpio.csv', delimiter=',')


df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)

df = df.drop_duplicates()

# Guardar el dataset limpio
df.to_csv('tu_archivo_limpio.csv', index=False)

print("Limpieza completada y guardada en 'tu_archivo_limpio.csv'")