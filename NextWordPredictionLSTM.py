# library haye mored niaz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Function vase load data
def load_data(file_path):
    with open(file_path, 'r') as f:
        #split kardan file be chandin word
        text = f.read().split(' ')
    return text

#load kardan data ba localpath
train_data = load_data(r'D:\University\Visual Studio Projects\NextWordPredictionUsingLSTM\ptb.trainreduced.txt')
valid_data = load_data(r'D:\University\Visual Studio Projects\NextWordPredictionUsingLSTM\ptb.valid.txt')
test_data = load_data(r'D:\University\Visual Studio Projects\NextWordPredictionUsingLSTM\ptb.test.txt')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)

# Convert kardan kalamat be integer
X_train = tokenizer.texts_to_sequences(train_data)
X_valid = tokenizer.texts_to_sequences(valid_data)
X_test = tokenizer.texts_to_sequences(test_data)
# gostaresh liste listha
X_train = [token for sublist in X_train for token in sublist]
X_valid = [token for sublist in X_valid for token in sublist]
X_test = [token for sublist in X_test for token in sublist]

# peyda kardan tedad unique wordha dar data
vocab_size = len(tokenizer.word_index) + 1

# function baraye tolid sequence
def create_sequences(text, sequence_length):
    sequences = []
    
    for i in range(sequence_length, len(text)):
        
        sequences.append(text[i-sequence_length:i+1])
    return sequences

# Tarif toole sequence
sequence_length = 10

# baraye har 3 data sequence misazim
sequences_train = create_sequences(X_train, sequence_length)
sequences_valid = create_sequences(X_valid, sequence_length)
sequences_test = create_sequences(X_test, sequence_length)

# hame kalamat joz akhari ro input va akhari ro output gharar midim
X_train, y_train = [seq[:-1] for seq in sequences_train], [seq[-1] for seq in sequences_train]
X_valid, y_valid = [seq[:-1] for seq in sequences_valid], [seq[-1] for seq in sequences_valid]
X_test, y_test = [seq[:-1] for seq in sequences_test], [seq[-1] for seq in sequences_test]

# sequence va output ro be array convert mikone
X_train, y_train = np.array(X_train), np.array(y_train)
X_valid, y_valid = np.array(X_valid), np.array(y_valid)
X_test, y_test = np.array(X_test), np.array(y_test)

# khorooji ha ro be one-hot vector tabdil mikonim
y_train = to_categorical(y_train, num_classes=vocab_size)
y_valid = to_categorical(y_valid, num_classes=vocab_size)
y_test = to_categorical(y_test, num_classes=vocab_size)

# tarif model
model = Sequential()
# ezafe kardan yek layer embedding
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=sequence_length))
# ezafe kardan yek layer LSTM
model.add(LSTM(100, return_sequences=True))
# ezafe kardan yek layer LSTM
model.add(LSTM(100))
# ezafe kardan Dense layer vase prediction
model.add(Dense(vocab_size, activation='softmax'))

# compile mikonim
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model ro train mikonim
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

# inam vase test model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')


#tooye run haye mokhtalef accuracy model taghriban 10-13% boode ke adad payinie
#vase bala bordan accuracy mishe hajm training data ro ziad kard ke be memory ziadi niaz dare
#tooye in train az Penn Treebank dataset estefade shode ke training datash chizi nazdik 900k kalame dasht
#ama be dalil kambood memory va inke faghat hadaf test model bood size in corpus ro reduce kardam be 100k word
#file training data asli tooye githubam mojoode be esme trainOriginal
#vase hal moshkel kambood memory mishe moghe train az loss dige e mesle sparse_categorical_crossentropy estefade kard ya az generator estefade kard ke kolle data yehooyi load nashe
#ama chon sardard dashtam jedi dige hale implement kardan khotoot balayi ro nadaram bebakhshid biaid be hamin accuracy razi bashim :(
#AlirezaTimas