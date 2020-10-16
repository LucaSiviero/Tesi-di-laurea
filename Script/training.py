#Python 2.7+. In questo file viene inizializzato e addestrato il modello sul dataset imdb_reviews/subwords8k
#Al momento della realizzazione di questo file stavo provando a caricare i pesi da un modello già addestrato, per questo viene effettuato un caricamento dei pesi di una singola epoca.
import tensorflow as tf 
import tensorflow_datasets as tfds
import os
#Importo il dataset
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
#Split tra training e testing sets
train_dataset, test_dataset = dataset['train'], dataset['test']


checkpoint_path = "../training/cp.ckpt"     #Checkpoint per prendere i pesi (dopo una sola epoca di training)
checkpoint_path2 = "../training/cp2.ckpt"   #Checkpoint per salvare i pesi dopo 9 epoche totali di training
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)    #Callback per recuperare i pesi

cp_callback2 = tf.keras.callbacks.ModelCheckpoint(checkpoint_path2, save_weights_only=True, verbose=1)

encoder = info.features['text'].encoder     #Attraverso l'encoder eseguo l'embedding delle features, in modo da trasformare le stringhe in numeri

BUFFER_SIZE = 10000     #Grandezza del buffer per lo shuffle dei dati di input
BATCH_SIZE = 64         #Intervalli di input per la rete
padded_shapes = ([None], ())    #Shape dei dati trattati

train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes = padded_shapes)  #Shuffle e reshape dei dati di training

test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes = padded_shapes)     #reshape dei dati di test

#Costruisco il modello
model = tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, 64),
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(1, activation='sigmoid')])
#Utilizzo model.compile() per preparare i parametri del training
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])

model.load_weights(checkpoint_path) #Carico i pesi di una singola epoca di training
history = model.fit(train_dataset, epochs = 8, validation_data = test_dataset, validation_steps=30, callbacks=[cp_callback2])   #Alleno il modello.
