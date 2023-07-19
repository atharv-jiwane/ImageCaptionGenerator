import numpy as np
from PIL import Image
import os
import string
from pickle import dump
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.text import Tokenizer #for text tokenization
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Dense#Keras to build our CNN and LSTM
from keras.layers import LSTM, Embedding, Dropout
from tqdm.notebook import tqdm as tqdm #to check loop progress
tqdm().pandas()
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def img_capt(filename):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split("\t")
        
        # take the first token as image id, the rest as description
        image_id, image_desc = tokens[0], tokens[1:]
        
        # extract filename from image id
        image_id = image_id.split('.')[0]
        
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        if image_id not in descriptions:
            descriptions[image_id] = list()
        descriptions[image_id].append(image_desc)
    
    return descriptions

# Data cleaning function will convert all upper case alphabets 
# to lowercase, removing punctuations and words containing numbers
def txt_clean(captions):
    table = str.maketrans('','',string.punctuation)
    for img, caps in captions.items():
        for i in range(len(caps)):
            descp = caps[i]
            descp = descp.split()
            #uppercase to lowercase
            descp = [word.lower() for word in descp]
            #remove punctuation from each token
            descp = [word.translate(table) for word in descp]
            #remove hanging 's and a
            descp = [word for word in descp if(len(word)>1)]
            #remove words containing numbers with them
            descp = [word for word in descp if(word.isalpha())]
            #converting back to string
            caps[i] = ' '.join(descp)
    
    return captions


def txt_vocab(descriptions):
  # To build vocab of all unique words
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab

#To save all descriptions in one file
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()

dataset_text = r"C:\Users\Atharv Jiwane\Downloads\Flickr8k_text\Flickr8k.token.txt"
dataset_images = r"C:\Users\Atharv Jiwane\Downloads\Flickr8k_Dataset\Flicker8k_Dataset"

filename = dataset_text
descriptions = img_capt(filename)
print("Length of descriptions =" ,len(descriptions))
#cleaning the descriptions
clean_descriptions = txt_clean(descriptions)
#to build vocabulary
vocabulary = txt_vocab(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))
#saving all descriptions in one file
save_descriptions(clean_descriptions, "descriptions.txt")


model = Xception( include_top=False, pooling='avg' )
def extract_features(directory):
    model = Xception( include_top=False, pooling='avg' )
    features = dict() 

    for pic in tqdm(os.listdir(directory)):
        file = directory + "/" + pic
        image = Image.open(file)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis = 0)
        image = image / 127.5
        image = image - 1.0
        feature = model.predict(image)
        features[pic] = feature
    
    return features

features = extract_features(dataset_images)

len(features)
dump(features, open("features.p", "wb"))

features  = load(open("features.p", "rb"))
print(features)
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos

filename = r"C:\Users\Atharv Jiwane\Downloads\Flickr8k_text\Flickr_8k.trainImages.txt"
# train = loading_data(filename)
train_imgs = load_photos(filename)

print(len(train_imgs))
doc = load_doc("descriptions.txt")
print(doc)
print(train_imgs)
def load_clean_descriptions(filename, photos):
    #loading clean_descriptions
    doc = load_doc(filename)
    descriptions = dict()

    for line in doc.split('\n'):
        words = line.split()
        if len(words) < 1:
            continue

        image_id, image_caption = words[0], words[1:]

        image_id_ext = image_id + ".jpg"

        if image_id_ext in photos:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            cap_gem = 'startseq ' + ' '.join(image_caption) + ' endseq' #this line is not executing
            descriptions[image_id].append(cap_gem)

    return descriptions


train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
print(train_descriptions)
def load_features(photos):
    #loading all features
    all_features = load(open("features.p","rb"))
    #selecting only needed features
    features = {k:all_features[k] for k in photos}
    return features




train_features = load_features(train_imgs)
#convert dictionary to clear list of descriptions
def dict_to_list(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc
#creating tokenizer class
#this will vectorise text corpus
#each integer will represent token
from keras.preprocessing.text import Tokenizer
def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer
# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
vocab_size 
# #The size of our vocabulary is 7577 words.
#calculate maximum length of descriptions to decide the model structure parameters.
def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)
    
max_length = max_length(descriptions)
print(max_length)
#  #Max_length of description is 32
def create_sequences(tokenizer, max_length, desc_list, feature):
    x_1, x_2, y = list(), list(), list()
    # move through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        print(len(seq))
        # divide one sequence into various X,y pairs
        for i in range(1, len(seq)):
            # divide into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            x_1.append(feature)
            x_2.append(in_seq)
            y.append(out_seq)
            
    return np.array(x_1), np.array(x_2), np.array(y)
def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            #retrieve photo features
            feature = features[key + ".jpg"][0]
            inp_image, inp_sequence, op_word = create_sequences(tokenizer, max_length, description_list, feature)
            yield [[inp_image, inp_sequence], op_word]
    

#To check the shape of the input and output for your model
[a, b], c = next(data_generator(train_descriptions, features, tokenizer, max_length))
a.shape, b.shape, c.shape
#((47, 2048), (47, 32), (47, 7577))
from keras.utils import plot_model
from keras.layers import concatenate

def define_model(vocab_size, max_length):
    # CNN
    inputs1 = Input(shape = (2048, ))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation = "relu")(fe1)
    # LSTM
    inputs2 = Input(shape = (max_length, ))
    se1 = Embedding(vocab_size, 256, mask_zero = True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # Merging both models
    decoder1 = concatenate([fe2, se3])
    decoder2 = Dense(256, activation = "relu")(decoder1)
    outputs = Dense(vocab_size, activation = "softmax")(decoder2)
    # merge it [image, seq] [word]
    model = Model(inputs = [inputs1, inputs2], outputs = outputs)
    model.compile(loss = "categorical_crossentropy", optimizer="adam")
    # summarize model
    print(model.summary())
    plot_model(model, to_file="model.png", show_shapes = True)

    return model


print("Dataset : ", len(train_imgs))
print("Descriptions : train = ", len(train_descriptions))
print("Photos : train = ", len(train_features))
print("Vocabulary Size : ", vocab_size)
print("Description Length : ", max_length)

model = define_model(vocab_size, max_length)
epochs = 10
steps = len(train_descriptions)


for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit_generator(generator, epochs = 1, steps_per_epoch= steps, verbose = 1)
    model.save("models/model_new_last_" + str(i) + ".h5")

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen = max_length)
        pred = model.predict([photo, sequence], verbose = 0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)

        if word is None:
            break

        in_text += ' ' + word

        if word == 'endseq':
            break

    return in_text

from nltk.translate.bleu_score import corpus_bleu


def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()

    for key, desc_list in descriptions.items():
        yhat = generate_desc(model, tokenizer, photos[key + ".jpg"], max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

    return
filename = r'C:\Users\Atharv Jiwane\Downloads\Flickr8k_text\Flickr_8k.testImages.txt'
test = load_photos(filename)

print("Dataset : ", len(test))

test[1]
test_descriptions = load_clean_descriptions("descriptions.txt", test)
print("Descriptions : ", len(test_descriptions))
test_features = load_features(test)
print("Photos : ", len(test_features))
model = load_model("models/model_new_last_9.h5")
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
img_path = r"C:\Users\Atharv Jiwane\Downloads\Flickr8k_Dataset\Flicker8k_Dataset\2677656448_6b7e7702af.jpg"

model_path = r"models\model_new_last_9.h5"

model = load_model(model_path)
xception_model = Xception(include_top=False, pooling="avg")
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)
description = generate_desc(model, tokenizer, photo, max_length)
print("nn")
print(description)
plt.imshow(img)
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Can't open image! Ensure that image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
        # for 4 channels images, we need to convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature
def beam_search_caption(image_path, beam_length=2, topn=5):
    image = np.expand_dims(extract_features(image_path),axis=0)
    beam = [ (1,'startseq') ]
    ans = []
    while len(beam)>0:
        cp, in_text = beam.pop(0)
        if len(in_text.split()) >= max_length:
            ans.append((cp,in_text))
            continue
        newwords  = getNextWords(image,in_text,n=beam_length)
        for nw,prob in newwords:
            if nw == "endseq":
                ans.append( (cp*prob,in_text+" "+nw) )
            else:
                beam.append( (cp*prob,in_text+" "+nw) )
    return [x[1] for x in sorted(ans,key=lambda x:x[0],reverse=True)[:topn]]
beam_search_caption('dev/dev1.jpeg')
