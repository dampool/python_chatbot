# -*- coding: utf-8 -*-
"""
Created on April 8, 2020

@author: mayod
"""

import nltk
import random  # used to generate random responses
import string  # used to remove punctuation
#nltk.download('wordnet')

#nltk.download('punkt')

with open("python_doc.txt", 'r',errors = 'ignore') as myFile1:
  data1 = myFile1.read()
#
# sent_tokens = data1
sent_tokens = nltk.sent_tokenize(data1)# converts to list of sentences
# print('______________________')
# print(sent_tokens)

# with open("twilightZoneOutput.txt", 'r') as myFile2:
#   data2 = myFile2.read()
#
# out_text = data2.split("@")

greetings = ["hello", "hi", "greetings", "sup", "what's up", "hey"]
greetingResponses = (["Greetings. I sense several heat signatures within this room.", 
                       "Hello. All my systems are nominal.", "Good afternoon. I am ready to assist.", 
                       "Hello.  Will there be anything else?"])

botID = "Pybot: "
user_name = "Can you please enter your name"
normalResponse = "Good day "

confusedResponse = "I am unable to understand your directive."

thanks = ["thanks", "thank you", "cool", "awesome"]
welcomeResponse = "You are most welcome.  I am now powering down...engaging turret defenses."

goodbyes = ["bye", "goodbye", "later", "lates", "cya", "cyas", "peace"]
goodbyeResponse = "Goodbye. I am now powering down...engaging turret defenses."
    
lemmer = nltk.stem.WordNetLemmatizer()  # used to consolidate different word forms


# returns cleaned list of consolidated tokens
def lemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]  

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

# different method for removing non-alphanumeric characters
def lemNormalize(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# checks to see if the input text matches one of the greeting_inputs.  If so,
# return one of the random greeting_responses.
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greetings:
            return random.choice(greetingResponses)
        
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  

def response(user_response):
    bot_response=''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=lemNormalize)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # print('*****tfidf*********')
    # print(tfidf)
    vals = cosine_similarity(tfidf[-1], tfidf)
    # print('+++++++++++vals+++++++++++++++')
    # print(vals)
    idx=vals.argsort()[0][-2]
    # print('???????????idx????????')
    # print(vals.argsort())
    # print(idx)
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf==0):
        bot_response=bot_response+ confusedResponse
        return bot_response
    else:

        bot_response = bot_response + sent_tokens[idx]
        return bot_response

# flag=True
# print("\n\n" + botID + normalResponse)

def chat(user):
    #return botID + normalResponse
    user_response = user
    user_response = user_response.lower()
    if user_response not in goodbyes:
        if user_response in thanks:
            flag = False
            #print(botID + welcomeResponse)
            return botID + welcomeResponse
        else:
            if (greeting(user_response) != None):
                #print(botID + greeting(user_response))
                return botID + greeting(user_response)
            else:
                #print(sent_tokens)
                #sent_tokens.append(user_response)
                #print(botID, end="")
                #print(response(user_response))
                return response(user_response)
                sent_tokens.remove(user_response)
    else:
        flag = False
        return botID + goodbyeResponse
# while(flag==True):
#     user_response = input()
#     user_response=user_response.lower()
#     if user_response not in goodbyes:
#         if user_response in thanks:
#             flag=False
#             print(botID + welcomeResponse)
#         else:
#             if(greeting(user_response)!=None):
#                 print(botID + greeting(user_response))
#             else:
#                 sent_tokens.append(user_response)
#                 print(botID ,end="")
#                 print(response(user_response))
#                 sent_tokens.remove(user_response)
#     else:
#         flag=False
#         print(botID + goodbyeResponse)