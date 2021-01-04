from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from create_conversion_dict import get_dict

from os import listdir
from os.path import isfile, join
from os import path

import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
import fastqreader
import numpy as np

import math
import itertools
import pickle

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def load_file_names():
    #Returns a list of files names when they exist in both converted and fastq folders.
    output = []

    #Get file names in converted folder
    mypath = "data\\converted"
    onlyfiles = [f[0:-1-3] for f in listdir(mypath) if isfile(join(mypath, f))]

    #See if the files in the converted folder exist in the fastq folder
    for f in onlyfiles:
        if path.exists("data\\fastq\\"+f+".fastq"):
            output.append(f)
        #end
    #end

    return output
#end


def load_count(file,name,num_seqs_per_file):
    #Uses the pre-converted kmer counts.
    x = []
    y = []

    f = open(file,"r")

    for line in f.readlines():
        x.append(line.split(',')[0:-1]) #To 2nd last position as last pos is \n char
        y.append(name)
        if len(x) >= num_seqs_per_file:
            break
        #end
    #end

    return {"Sequence":x,"Location":y}
#end

def convert_seq(seq,char2num,k):
    #Converts the input sequence to ints then splits into kmers for input.
    converted = []
    for char in seq:
        #Checks only values in dict are used
        if char in char2num.keys():
            converted.append(tf.one_hot(char2num.get(char),4))
        #end
    #end


    return converted
#end

def load_LSTM(f,name,k,encoder,num_seqs_per_file):
    #Loads all data at once, ownt work for huge files ect.
    #folder is path to folder, lines is number of liens to read from each file.
    #output will be number of files in fodler * number of lines
    char2num = {
        "A":0,
        "T":1,
        "C":2,
        "G":3
    }
    #Number of sequences to read in
    num_lines = num_seqs_per_file

    x = []
    y = []

    #Read in list of sequences
    sequences = fastqreader.read("data\\fastq\\"+f+".fastq")
    curr_line = 1
    for seq in sequences:
        temp = np.array(convert_seq(seq.get("sequence",""),char2num,k))

        #Convert sequences using lstm model
        g = encoder.predict(tf.convert_to_tensor(np.reshape(temp,(1,temp.shape[0],4))))
        x.append(g)
        y.append(name)

        #Only load this number of sequences
        if curr_line >= num_lines:
            break
        #end

        curr_line+=1
    #end

    #Return
    return {"Sequence":x,"Location":y}
#end

def load_dhanjays(file,name):

    #load pickle dict
    file = open(file, 'rb')
    data = pickle.load(file)

    # dict_keys(['info', 'family_names', 'num_families', 'family_sizes', 'sequence_vectors'])

    # think can just give this to the prediction models and it should be fine
    return data.get("sequence_vectors")   
#end

def load_model(path):
    #Does what it says on the tin
    return keras.models.load_model(path)
#end

def create_forest_LSTM(train_data,name):
    #Create a random forest classifier with the specified data.

    train_percentage = 0.7
    estimators = 100

    forest=RandomForestClassifier(n_estimators=estimators)

    #Training
    for f in train_data:
        # print(f.get("Sequence"))
        hold_out_amount = math.floor(train_percentage*len(f.get("Sequence")))

        t = np.array(f.get("Sequence")[0:hold_out_amount]).shape

        forest.fit(np.reshape(f.get("Sequence")[0:hold_out_amount],(t[0],t[2])),np.reshape(f.get("Location")[0:hold_out_amount],(t[0])))
    #end

    #Predicting
    pred = []
    truth = []
    for f in train_data:
        hold_out_amount = math.floor(train_percentage*len(f.get("Sequence")))
        t = np.array(f.get("Sequence")[hold_out_amount:-1]).shape
        g = np.reshape(f.get("Sequence")[hold_out_amount:-1],(t[0],t[2]))
        pred.append(forest.predict(g).tolist())
        truth.append(f.get("Location")[hold_out_amount:-1])
    #end

    #Join pred and truth into long lists instead of list of lists for each file
    pred = list(itertools.chain.from_iterable(pred))
    truth = list(itertools.chain.from_iterable(truth))

    #Print accuracy of predictions
    print(name +" Accuracy for test percentage of "+ str(train_percentage) + " hold out is " + str(accuracy_score(truth, pred)))

    return forest
#end

def create_forest_Count(train_data,name):
    #Create a random forest classifier with the specified data.

    train_percentage = 0.7
    estimators = 100

    forest=RandomForestClassifier(n_estimators=estimators)

    #Training
    for f in train_data:
        hold_out_amount = math.floor(train_percentage*len(f.get("Sequence")))

        forest.fit(f.get("Sequence")[0:hold_out_amount],f.get("Location")[0:hold_out_amount])
    #end

    #Predicting
    pred = []
    truth = []
    for f in train_data:
        hold_out_amount = math.floor(train_percentage*len(f.get("Sequence")))
        pred.append(forest.predict(f.get("Sequence")[hold_out_amount:-1]))
        truth.append(f.get("Location")[hold_out_amount:-1])
    #end

    #Join pred and truth into long lists instead of list of lists for each file
    pred = list(itertools.chain.from_iterable(pred))
    truth = list(itertools.chain.from_iterable(truth))

    #Print accuracy of predictions
    print(name + " Accuracy for test percentage of "+ str(train_percentage) + " hold out is " + str(accuracy_score(truth, pred)))

    return forest
#end

def create_forest_Dhanjays(train_data,name):

#end



if __name__ == "__main__":
    #Needs to be in both converted and fastq folders
    files = load_file_names()

    #Conversion dictionary for location name
    conv_dict = get_dict()

    #Number of sequecnes to laod per file.
    num_seqs_per_file = 100

    ########################################################
    #Count Section
    ########################################################

    #Read in data from the kmer counts method.
    # count_data = []
    # for f in files:
    #     count_data.append(load_count("data/converted/"+f+".txt", conv_dict.get(f[0:3]),num_seqs_per_file))
    # #end

    ########################################################
    #LSTM Section
    ########################################################

    #Load pretrained LSTM model
    # encoder = load_model("encoder_model/")
    # encoder.compile(optimizer="adam",loss='mse')
    # encoder.summary()
    # k = 6

    # #Read in data from the LSTM method
    # LSTM_data = []
    # for f in files:
    #     LSTM_data.append(load_LSTM(f, conv_dict.get(f[0:3]), k, encoder,num_seqs_per_file))
    # #end
    

    ########################################################
    #Dhanjays Section
    ########################################################
    load_dhanjays("dhanjays/compiled_embeddings.pkl","Dhanjays")

    ########################################################
    #Random Forest Section
    ########################################################
    # count_forest = create_forest_Count(count_data,"Count")

    # LSTM_forest = create_forest_LSTM(LSTM_data,"LSTM")
#ends