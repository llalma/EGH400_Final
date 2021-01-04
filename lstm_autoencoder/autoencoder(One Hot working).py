# lstm autoencoder recreate sequence
import numpy as np
import tensorflow as tf
import fastqreader
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.preprocessing as preprocessing
import os
from keras import backend as K
import random as rand
from datetime import datetime
import matplotlib.pyplot as plt
import itertools

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def repeat(x):
    stepMatrix = K.ones_like(x[0][:,:,:1]) #matrix with ones, shaped as (batch, steps, 1)
    latentMatrix = K.expand_dims(x[1],axis=1) #latent vars, shaped as (batch, 1, latent_dim)

    return K.batch_dot(stepMatrix,latentMatrix)
#end

def create_model(k,input_dim):
    latent_dim = 128
    step_size = None #Needs to be none to allow for variable lenth input


    inputs = keras.Input(shape=(input_dim, 5))

    encoded = layers.Bidirectional(layers.LSTM(latent_dim),name="encoder")(inputs)
    
    decoded = layers.Lambda(repeat)([inputs,encoded])
    # decoded = layers.Reshape((1,latent_dim*2))(encoded)
    decoded = layers.LSTM(5, return_sequences=True,name="decoder")(decoded)
    # decoded = layers.Reshape((input_dim,1))(decoded)

    sequence_autoencoder = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)

    sequence_autoencoder.compile(optimizer="adam",loss='mse',metrics=['accuracy'])

    return sequence_autoencoder,encoder
#end    

def load_data(folder,num_lines,k):
    #Loads all data at once, ownt work for huge files ect.
    #folder is path to folder, lines is number of liens to read from each file.
    #output will be number of files in fodler * number of liness
    char2num = {
        "A":1,
        "T":2,
        "C":3,
        "G":4
    }
    #0 is padding value

    #########################################################################################
    #Real Data
    #########################################################################################
    #Find all files that end with fastq in folder
    files = []
    data = []
    for file in os.listdir(folder):
        if file.endswith(".fastq"):
            curr_line = 1
            sequences = fastqreader.read(os.path.join(folder, file))

            for seq in sequences:
                
                #Converted form
                data.append(np.array(convert_seq(seq.get("sequence",""),char2num,k)))

                if curr_line >= num_lines:
                    break
                #end

                curr_line+=1
            #end
        #end
    #end

    #Randomly shuffle data
    rand.shuffle(data)

    #Pad data all to the same length, 
    data = tf.one_hot(preprocessing.sequence.pad_sequences(data, maxlen=None, dtype='int32', padding='pre', truncating='pre',value=0.0),5)

    return np.array(data)
#end

def convert_seq(seq,char2num,k):
    #Converts the input sequence to ints then splits into kmers for input.
    converted = []
    for char in seq:
        #Checks only values in dict are used
        if char in char2num.keys():
            converted.append(char2num.get(char))
        #end
    #end

    #Trim data so it is divisiable by k
    while len(converted) % k != 0:
        converted = converted[0:-1]
    #end

    return converted
#end

def save_losses(name,iteration,loss):
    #Open the specified text file and write iteration number and loss to file
    f = open("logs/"+name,'a')
    f.write(str(iteration) + "," + str(loss) +"\n")
    f.close
#end

def plot_losses(file_name):
    f = open("logs/"+file_name,'r')
    x = []
    y = []

    for line in f.readlines() :
        temp = line.split(",")
        x.append(int(temp[0]))
        y.append(float(temp[1][0:-1-1]))
    #end
    
    plt.plot(x, y, label="Loss")
    plt.show()
#end

def get_accuracy(data,autoencoder):
    
    score = 0
    checked = 0
    preds = autoencoder.predict(data)

    

    for pred,d in zip(preds,data):
        pred = tf.argmax(pred, axis=1).numpy()
        d = tf.argmax(d, axis=1).numpy()

        if (pred == d).all():
            score+=1
        #end
        checked+=1
    #end

    # for x1 in data:
    #     pred = autoencoder.predict(x1).flatten()
    #     x1 = x1.flatten()

    #     f2d = {"0.027722917":0,
    #         "0.123689204":1,
    #         "0.23706964":2,
    #         "0.30790702":3,
    #         "0.34167075":4
    #     }

    #     output = []
    #     for i,p in enumerate(pred):
    #         output.append(f2d.get(str(p)))
    #     #end

    #     pred = output

    #     # print(pred)
    #     # print(x1)

    #     # for p,r in zip(pred,x1):
    #     #     if p == r:
    #     #         score+=1
    #     #     #end
    #     #     checked+=1
    #     # #end

    #     if (pred == x1).all():
    #         score+=1
    #     #ned
    #     checked+=1
    # #end

    print("score: " + str(score))
    print("checked: " + str(checked))
    return score/checked
#end

def save_model_auto(model):
    #Does what it says on the tin
    model.save_weights("autoencoder_model/")
#end 

def save_model_encoder(model):
    #Does what it says on the tin
    model.save("encoder_model/")
#end 

def load_model(path):
    #Does what it says on the tin
    return keras.models.load_model(path)
#end

if __name__ == "__main__":
    k = 6
    num_sequences = 10000
    epochs = 10
    step_size = None

    #Create a file to save the training losses
    log_name = datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
    open("logs/"+log_name,"w")

    #load in data
    data = load_data("data/",num_sequences,k)

    print(data[0])

    #Create LSTM Autoencoder
    autoencoder,encoder = create_model(k,len(data[0]))    #Create new model from scratch
    autoencoder.summary()

    autoencoder.load_weights("autoencoder_model/")  #load previous model

    # history = autoencoder.fit(x=data,y=data,epochs=epochs)

    #Train autoencoder
    # for i,seq in enumerate(data):
    #     history = autoencoder.fit(x=seq,y=seq,epochs=epochs)
        
    #     #Save loss
    #     if i % 100 == 0:
    #         print("Saving Logs")
    #         save_losses(log_name,i,history.history.get("loss")[0])
    #     #end
    # #end

    # print(seq)
    # print(tf.convert_to_tensor(np.array(autoencoder.predict(x=seq)).flatten()))


    #Save Autoencoder
    save_model_auto(autoencoder)

    #Save Encoder model
    encoder.load_weights("autoencoder_model/")
    save_model_encoder(encoder)

    #Get accuracy of the model
    print("Accuracy: " + str(get_accuracy(data,autoencoder)))

    plot_losses(log_name)


#end