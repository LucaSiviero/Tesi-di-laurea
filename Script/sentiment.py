#Requisiti: Python 2.7+, PortAudio, PyAudio, SpeechRecognition, TF, TFdatasets
import tensorflow as tf 
import tensorflow_datasets as tfds
import os
import speech_recognition as sr
import tensorflow
import numpy 
from numpy import savetxt

r = sr.Recognizer()             #Inizializzo il riconoscitore per ascoltare.
m = sr.Microphone()             #Inizializzo il microfono come fonte per il riconoscitore.

output_list = list()            #Inizializzo una lista vuota

try:
    print("A moment of silence, please...")
    with m as source: r.adjust_for_ambient_noise(source)
    print("Set minimum energy threshold to {}".format(r.energy_threshold))
    while True:                 #In un ciclo infinito ascolto l'input
        print("Say something!")
        with m as source: audio = r.listen(source)
        print("Got it! Now to recognize it...")
        try:
            # Riconosco il parlato attraverso il metodo recognize_google, che utilizza la tecnologia Google Cloud Speech API
            value = r.recognize_google(audio)

            # Differenzio due casi per scrivere il testo con la codifica corretta
            if str is bytes:                                                # Python 2 usa i byte per le stringhe
                print(u"You said \"{}\"".format(value).encode("utf-8"))
                output_list.append(format(value).encode("utf-8"))	    # Colloco nell'ultima posizione della lista la frase appena pronunciata
            else:                                                           # Python 3 usa unicode per le stringhe
                print("You said {}".format(value))
        except sr.UnknownValueError:                                        # Eccezione lanciata quando non si riesce a trascrivere l'input
            print("Oops! Didn't catch that")
        except sr.RequestError as e:                                        # Eccezione lanciata quando si hanno problemi con Google Cloud Speech API
            print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
except KeyboardInterrupt:                                                   # Interruzione dell'ascolto con Ctrl+C
    pass

string = " ".join(output_list)      #Concateno il contenuto della lista, separando le singole frasi con uno spazio. string sara' usato come input per la rete neurale.
file = open("../output.csv", 'w')   #Ottengo un riferimento al file su cui voglio salvare le recensioni
savetxt(file, output_list, delimiter=';', fmt='%s', newline=".")    #Salvo le recensioni nel file, con il punto invece di \n, altrimenti vado a capo ogni volta che dico una frase

#L'intero dataset deve essere importato per far si che il modello abbia familiarita' con l'embedding del dataset, realizzato dall'encoder.
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)    #Accedo al dataset attraverso tfds. Se non e' presente nel sistema, viene scaricato automaticamente.

checkpoint_path = "../training/cp2.ckpt"                            #Ottengo un riferimento ai pesi salvati dopo 9 epoche di training.
checkpoint_dir = os.path.dirname(checkpoint_path)

encoder = info.features['text'].encoder         #Utilizzo l'encoder per realizzare l'embedding delle features.

#Definisco la struttura del modello:
model = tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, 64),         
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(1, activation='sigmoid')])

#Carico i pesi del training
model.load_weights(checkpoint_path)

def pad_to_size(vec, size):     #Con questo metodo eseguo un padding di zeri sulla sentenza che la rete riceve in input per il testing.
    zeros = [0] * (size-len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence, pad):      #Qui preparo la sentenza come input per la rete
    encoded_sample_pred_text = encoder.encode(sentence)         #Realizzo l'embedding della frase in input
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)        #Padding di zeri
        encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)    #Cast a float dopo l'embedding e il padding.
        predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))    #Model.predict() per ottenere la probabilita' con cui la sentenza in input e' positiva.
    return predictions

predictions = sample_predict(string, pad=True)          #Predictions e' un float che rappresenta la probabilita' con cui la recensione data in input e' positiva

if predictions < 0.45:          #Se la probabilita' e' minore di 0.45, la recensione e' negativa
    print("Negative review")

if(predictions >= 0.45):        #Altrimenti e' positiva.
    print("Positive review")
