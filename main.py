import nltk

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle
import requests
import pprint

with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []  # putting every word from the "patterns" in this array
    labels = [] # having all the tags in the labesl array
    doc1 = [] # putting every pattern list/array in this array
    doc2 = [] # for each entry(pattern) in doc_x there will be the corresponding pattern for dox_y/ this is important when trining the modole so we know how to classfiy

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            #return a list or array with all the words in the pattern array in the json file
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            doc1.append(wrds)
            doc2.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] # converting all of the upper case letters to lower case
    words = sorted(list(set(words))) # removing all of the duplicates so we know what vocab the bot has been exposed to and sorting the words

    labels = sorted(labels)

    training = [] # all the data in the intents folder
    output = [] # all the tags for the specific intent

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(doc1):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(doc2[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
      pickle.dump((words, labels, training, output), f)

#making the modle
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
#hiddne layers each is 8 nerurons
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
#output layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
    model.save("Model.tflearn")

def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s in s_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

                
    return numpy.array(bag)

def chat():

    print("How can i help you!!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
               responses = tg['responses']
               print(tag)
               if tag == "weather":
                   r = requests.get('http://api.openweathermap.org/data/2.5/weather?id=292223&appid=ee00bb64d0d638dee83810e25c1d9651').json()
                   description = r['weather'][0]['description']
                   temp = r['main']['temp']
                   print("it looks to be " + description + " and the temptrure is " + str(int(temp - 273.15)))
               elif tag == "assignment":
                   print("hello assignment")
               elif tag == "articles":
                   inp = input("what is the topic about ")
                   i=0

                   url = "https://core.ac.uk/api-v2/articles/search/"+ inp +"?page=1&pageSize=10&metadata=true&fulltext=false&citations=false&similar=false&duplicate=false&urls=false&faithfulMetadata=false&apiKey=H1UX6zVBpWb0FTjAG45Jl2nZYe7gRucr"
                   response = requests.get(url).json()
                   while i<= 3:
                     a = response['data'][i]['authors']
                     b = response['data'][i]['description']
                     c = response['data'][i]['downloadUrl']
                     print("The article is written by " + str(a) + "\n and here is a brief description --> " + b + "\n"+ c )
                     i += 1
               elif tag == "covid":
                   url = "https://covid-19-data.p.rapidapi.com/totals"

                   headers = {
                       'x-rapidapi-key': "ca6f9029e8mshc3227e6f2ce2113p16b2dfjsn75d6cdaefbd6",
                       'x-rapidapi-host': "covid-19-data.p.rapidapi.com"
                   }

                   response = requests.request("GET", url, headers=headers).json()
                   confirmed_cases = response[0]['confirmed']
                   recovered_cases = response[0]['recovered']
                   death = response[0]['deaths']



                   print(" the number of confirmed cases in the world is " + str(confirmed_cases) + " and the number of recovered cases is " + str(recovered_cases) + " the total number of deaths world wide is " + str(death))
                   print(response)
               else:
                   print(random.choice(responses))




chat()





