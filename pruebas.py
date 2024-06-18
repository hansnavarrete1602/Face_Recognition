import cv2
import requests
import speech_recognition as sr
import opuslib

# Crear el objeto VideoCapture con la dirección IP de la cámara
camara = cv2.VideoCapture("http://192.168.0.7:8080/video")
# Crear el objeto Recognizer
reconocedor = sr.Recognizer()

# Hacer un bucle infinito
while True:
    # Leer el frame actual de la cámara
    ret, frame = camara.read()
    # Mostrar el frame en una ventana
    cv2.imshow("Camara IP", frame)
    # Obtener el audio de la cámara
    respuesta = requests.get("http://192.168.0.7:8080/audio.opus")
    # Comprobar el estado de la respuesta
    if respuesta.status_code == 200:
        # Crear un objeto OpusDecoder
        decodificador = opuslib.OpusDecoder(48000, 1)
        # Decodificar el audio .opus a PCM
        pcm = decodificador.decode(respuesta.content, 960, False)
        # Reconocer el texto del audio usando Google
        try:
            # Usar el idioma español
            texto = reconocedor.recognize_google(pcm, language="es-ES")
            # Mostrar el texto reconocido en la consola
            print("Has dicho: " + texto)
        except sr.UnknownValueError:
            # Mostrar un mensaje de error si no se pudo reconocer el audio
            print("No se pudo reconocer el audio")
        except sr.RequestError as e:
            # Mostrar un mensaje de error si hubo un problema al llamar al servicio de Google
            print("Error al llamar al servicio de Google; {0}".format(e))
    # Esperar a que se presione la tecla q para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar el objeto VideoCapture
camara.release()
# Destruir todas las ventanas
cv2.destroyAllWindows()
