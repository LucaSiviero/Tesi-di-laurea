Nella cartella "Pesi" ci sono i checkpoint salvati.

Nella cartella "Script" ci sono due file:

-sentiment.py è l'intero progetto funzionante, riceve in input delle frasi pronunciate nel microfono e queste vengono salvate in un file .CSV.
 Successivamente, la frase viene analizzata dalla rete per fare una classificazione positiva o negativa.
-training.py è il file che ho compilato per addestrare la rete. Richiede un checkpoint iniziale da cui carico la prima epoca di training (perché ho eseguito il caricamento dei pesi di una epoca di training precedente).
Tutti i checkpoint sono salvati in "Pesi".
La SpeechToText è implementata attraverso il package "SpeechRecognition", che richiede l'installazione di portaudio e pyaudio, oltre che del package stesso "speechrecognition".Inoltre, richiede una connessione ad internet per utilizzare il servizio "Google Cloud Speech API".
Il modello utilizza tensorflow con back-end keras, ed è addestrato sul dataset "imdb_reviews/subwords8k", che non includo nella cartella per due motivi:
1-è un po' grande;
2-Viene scaricato in automatico da tensorflow datasets nella cwd del file.

