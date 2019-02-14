import keras, pickle, requests, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, url_for, redirect, request, session

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/lyrics', methods=['POST'])

def generate_lyrics(seq_len = 4,
                    song_len = 250,
                    temperature = 1.2):
    
    with open('../models/cLSTM315-1015/1549314666_LSTM350_1015tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
           
    model = keras.models.load_model('../models/cLSTM315-1015/1549314666_LSTM315_1015model.h5')
         
    seed = request.form['seed']
    
    if not seed:
        return redirect(url_for('index'))
    
    seed_clean = seed.lower().split(' ')
    print(seed_clean)
    doc = []
    
    while len(doc) < song_len:
        text = [seed_clean]
        sequence = [tokenizer.texts_to_sequences([word])[0] for word in text]
        pad_sequence = pad_sequences(sequence, maxlen=seq_len, truncating='pre')
        sequence_reshape = np.reshape(pad_sequence, (1, seq_len))

        yhat = model.predict(sequence_reshape, verbose=0)[0]
            
        preds = np.asarray(yhat).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)

        next_index = np.argmax(probas)
        
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                seed_clean.append(word)
                doc.append(word)

    result = ' '.join(doc)
    result = result.split('\n')
    
    session['result'] = result
    session['tokenizer'] = tokenizer 
    session['model'] = model
    
    return render_template('lyrics.html', result=result, seed=seed)
    
@app.route('/save', methods=['GET','POST'])
    
def save_lyrics(name, email, seed, song):
    name = request.form['name']
    name = name.lower().strip()
    email = request.form['email']
    seed = request.form['seed']
    song = session.get('result', None)
    
    song += f'\n\n name: {name} \n email: {email} \n seed: {seed}'    
    
    now = round(time.time())
    with open(f'songs/{name}_{now}.txt', "w") as f:
        print(song, file=f)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)