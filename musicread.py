import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


notes = []


for file in glob.glob("midi_songs/*.mid"):
    print(file)
    midi = converter.parse(file)
    notes_to_parse = None;

    parts = instrument.partitionByInstrument(midi)

    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

with open('data/notes', 'wb') as filepath:
    pickle.dump(notes, filepath)

print("notes:",notes)

#***********************#
n_vocab = len(set(notes))
#***********************#

# every 100 notes to predict the next note
sequence_length = 100

# get all pitch names
pitchnames = sorted(set(item for item in notes))
print("pitchnames:",pitchnames)

# map pitch and integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
print("note_to_int:",note_to_int)

network_input = []
network_output = []

# create input sequences and corresponding outputs
#0.4对应对是2，就是距离第一个数字的距离是2
for i in range(0, len(notes)-sequence_length, 1):
    sequence_in = notes[i:i+sequence_length]
    sequence_out = notes[i+sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

print("network_input: ",network_input)
print("network_output: ",network_output)
n_patterns = len(network_input)

# reshape input to LSTM layers format
network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
print("network_input: ",network_input)

# normalize input
network_input = network_input / float(n_vocab)
#
network_output = np_utils.to_categorical(network_output)

model = Sequential()
model.add(LSTM(512,
               input_shape=(network_input.shape[1], network_input.shape[2]),
               return_sequences=True
               ))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add((Dense(256)))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# save weighted in file
#filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor= 'loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]

model.fit(network_input, network_output, epochs=20, batch_size=64, callbacks=callbacks_list)



