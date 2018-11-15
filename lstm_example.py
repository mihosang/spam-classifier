import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

max_words = 1000
max_len = 150

def RNN():
    global max_len, max_words
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

def import_phishing_data(paths):
    if len(paths) == 0:
        print("ERROR: Empty input file paths")
        return pd.DataFrame()
    dataset = None
    for path in paths:
        data = pd.read_json(path, encoding='utf-8')
        data['label'] = data.phishing.map({True: 1, False: 0})
        data = data.drop(["audioId"], axis=1)
        # data = data.take(np.random.permutation(len(data))[:1000])
        print("len(data) = %s" % len(data))
        if dataset is None:
            dataset = data
        else:
            dataset = pd.concat([dataset, data],sort=False)
    return dataset

if __name__ == "__main__":
    model = RNN()
    # STEP 1. Import & Prepare dataset
    data = import_phishing_data([
        "./mldataset/phishing/abnormal/phishing-2018-10-18-161054.json",
        "./mldataset/phishing/abnormal/phishing-meta-2018-10-23-115644.json",
        "./mldataset/phishing/normal/logo.csv-AsBoolean.json",
        "./mldataset/phishing/normal/normal-2018-10-18-210902.json"
    ])
    # STEP 2. Split into Train & Test sets
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.3, random_state=10)

    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
    print(sequences_matrix.shape)
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.fit(sequences_matrix, y_train, batch_size=128, epochs=10,
              validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

    accr = model.evaluate(test_sequences_matrix,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))