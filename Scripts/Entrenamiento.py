import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("./Datos/dataset_limpio.csv", delimiter=",", quotechar='"', dtype=str, on_bad_lines="skip")

print(df)

X = df["Descripción"]  
y = df["Tipo"]  


vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


joblib.dump(model, "./Modelos/chatbot_model.pkl")
joblib.dump(vectorizer, "./Modelos/tfidf_vectorizer.pkl")


print(f"Entrenamiento completado...........{accuracy:.2%}")
print("\nRespose:")
print(classification_report(y_test, y_pred))
