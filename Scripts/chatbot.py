import pandas as pd
import joblib

model = joblib.load("./Modelos/chatbot_model.pkl")
vectorizer = joblib.load("./Modelos/tfidf_vectorizer.pkl")

df = pd.read_csv("./Datos/dataset_limpio.csv", delimiter=",", quotechar='"', dtype=str, on_bad_lines="skip")
ón
respuesta_dict = dict(zip(df["Descripción"], df["Respuesta"]))

def clasificar_y_responder(consulta):
 
    consulta_tfidf = vectorizer.transform([consulta])

    tipo_predicho = model.predict(consulta_tfidf)[0]

    respuesta = respuesta_dict.get(consulta, "Lo siento, no encontré una respuesta específica para tu consulta.")

    return f"Clasificación: {tipo_predicho}\nRespuesta: {respuesta}"

# Ejemplo de uso 
while True:
    consulta_usuario = input("\n Ingresa tu consulta (o escribe 'salir' para terminar): ")
    if consulta_usuario.lower() == "salir":
        print("¡Hasta luego..........!")
        break
    respuesta_chatbot = clasificar_y_responder(consulta_usuario)
    print("\n Chatbot:", respuesta_chatbot)
