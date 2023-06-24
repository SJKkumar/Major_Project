'''
a=input().split(',')
b=input().split(',')
print(len(a),len(b))
for i,j in zip(a,b):
    print(i+": "+j)
'''

#1,0,30,3,0.6,0.6,9,960,2,2,0,1,0,0,0,0,0,0,1,0,0,0,0,0 --> not a pedo
#32,0.71875,41.28125,445,0.710353769,0.778967503,15109,78840,2,0,2,30,2,95,4,0,91,0.038,0.815,0.147,0.9999,137,42,1 --> a pedo

#number of conversation,percent of conversations started by the author,difference between two preceding lines in seconds,number of messages sent,average percent of lines in conversation,average percent of characters in conversation,number of characters sent by the author,mean time of messages sent,number of unique contacted authors,avg number of unique authors interacted with per conversation,total unique authors and unique per chat difference,conversation num and total unique authors difference,average question marks per conversations,total question marks,total author question marks,avg author question marks,author and conversation quetsion mark differnece,author total negative in author conv,author total neutral in author conv,author total positive in author conv, authortotal compound in author conv,pos word count author,neg word count author,prof word count author
'''
## not a pedo

number of conversation: 1
percent of conversations started by the author: 0
difference between two preceding lines in seconds: 30
number of messages sent: 3
average percent of lines in conversation: 0.6
average percent of characters in conversation: 0.6
number of characters sent by the author: 9
mean time of messages sent: 960
number of unique contacted authors: 2
avg number of unique authors interacted with per conversation: 2
total unique authors and unique per chat difference: 0
conversation num and total unique authors difference: 1
average question marks per conversations: 0
total question marks: 0
total author question marks: 0
avg author question marks: 0
author and conversation quetsion mark differnece: 0
author total negative in author conv: 0
author total neutral in author conv: 1
author total positive in author conv: 0
 authortotal compound in author conv: 0
pos word count author: 0
neg word count author: 0
prof word count author: 0

## a pedo

number of conversation: 32
percent of conversations started by the author: 0.71875
difference between two preceding lines in seconds: 41.28125
number of messages sent: 445
average percent of lines in conversation: 0.710353769
average percent of characters in conversation: 0.778967503
number of characters sent by the author: 15109
mean time of messages sent: 78840
number of unique contacted authors: 2
avg number of unique authors interacted with per conversation: 0
total unique authors and unique per chat difference: 2
conversation num and total unique authors difference: 30
average question marks per conversations: 2
total question marks: 95
total author question marks: 4
avg author question marks: 0
author and conversation quetsion mark differnece: 91
author total negative in author conv: 0.038
author total neutral in author conv: 0.815
author total positive in author conv: 0.147
 authortotal compound in author conv: 0.9999
pos word count author: 137
neg word count author: 42
prof word count author: 1
'''

'''
eg:
pedo:hi[0]--pedo:hello, there.[7]--kid:hii[47]--pedo:how old are you[70]--kid:10, u???[90]--pedo:nice[94]--pedo:so, where do you study[112]--kid:at school, duh.[150]--pedo:a sexy reply[160]--pedo:which school??[172]--pedo:where is it?[180]--kid:charles, why???[200]--pedo:i studied there too..[217]--kid:oh[230]--pedo:yes[239]--pedo:u know miss agnes[280]--pedo:she was my favorite teacher[290]--pedo:she taught some good values[301]--kid:i don't know her[340]--pedo:i will introduce you to her[356]--pedo:when shall we meet[367]--kid:no thnx[390]--kid:who are u btw??[426]
'''

##import numpy as np##import pandas as pd
##from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
##analyser = SentimentIntensityAnalyzer()
##from profanity_check import predict_prob
##from flask import Flask, request, jsonify, render_template
##import pickle
##
##app = Flask(__name__)
##model = pickle.load(open('C:\Users\vvssa\Pedophile project\model.pkl', 'rb'))
##
##@app.route('/')
##def home():
##    return render_template('index.html')
##
##@app.route('/predict',methods=['POST'])
##def predict():
##    def print_sentiment_scores(tweets):
##        vadersenti = analyser.polarity_scores(tweets)
##        return pd.Series([vadersenti['pos'], vadersenti['neg'], vadersenti['neu'], vadersenti['compound']])
##
##    int_features = request.form.values()[0].split()
##    final_features = [np.array(int_features)]
##    prediction = model.predict(final_features)
##
##    output = round(prediction[0], 2)
##
##    if output<0.5:return render_template('index.html', prediction_text='He is not suspected as a pedophile')
##    else:return render_template('index.html', prediction_text='He is suspected as a pedophile')
##@app.route('/results',methods=['POST'])
##def results():
##
##    data = request.get_json(force=True)
##    prediction = model.predict([np.array(list(data.values()))])
##
##    output = prediction[0]
##    return jsonify(output)
##
##if __name__ == "__main__":
##    app.run(debug=True)

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from profanity_check import predict_prob
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("C:\\Users\\yashvanth\\Desktop\\pedophile project\\svmmodel.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    def print_sentiment_scores(tweets):
        vadersenti = analyser.polarity_scores(tweets)
        return pd.Series([vadersenti['pos'], vadersenti['neg'], vadersenti['neu'], vadersenti['compound']])

    messages=[]
    for i in request.form.values():
        messages.extend(i.split('--'))
    feed_data=[len(messages)] #number of conversation
    feed_data.append(0.75) #percent of conversations started by the author
    avgsec=0
    meantime=0
    for i in messages:
        avgsec+=int(i[i.index('[')+1:i.index(']')])
    meantime=avgsec
    avgsec/=len(messages)
    diffsec=0
    t=0
    for i in messages[1:]:
        diffsec+=int(i[i.index('[')+1:i.index(']')])-t
        t=int(i[i.index('[')+1:i.index(']')])
    diffsec/=len(messages)
    feed_data.append(diffsec) #difference between two preceding lines in seconds
    feed_data.append(len(messages)) #number of messages sent
    nol=0
    for i in messages:
        if len(i)>1:nol+=1
    nol/=len(messages)
    feed_data.append(nol) #average percent of lines in conversation
    qp=0
    for i in messages:
        qp+=i.count('?')/len(i)
    qp/=len(messages)
    feed_data.append(nol-qp) #average percent of characters in conversation
    autchar=0
    qmark=0
    aqmark=0
    acount=0
    for i in messages:
        qmark+=i.count('?')
        if 'pedo' in i:
            acount+=1
            aqmark+=i.count('?')
            autchar+=len(i)-5
    totqmark=qmark
    qmark/=len(messages)
    feed_data.append(autchar) #number of characters sent by the author
    feed_data.append(meantime) #mean time of messages sent
    feed_data.append(2) #number of unique contacted authors
    feed_data.append(0) #avg number of unique authors interacted with per conversation
    feed_data.append(2) #total unique authors and unique per chat difference
    feed_data.append(30) #conversation num and total unique authors difference
    feed_data.append(qmark) #average question marks per conversations
    feed_data.append(totqmark) #total question marks
    feed_data.append(aqmark) #total author question marks
    feed_data.append(aqmark/acount) #avg author question marks
    feed_data.append(totqmark-aqmark) #author and conversation quetsion mark differnece




    pos=neg=neu=comp=0
    wp=wn=wnu=wc=0
    for i in messages:
        if i[0]=='p':
            tpos,tneg,tneu,tcomp=print_sentiment_scores(i[5:])
            pos+=tpos
            wp+=int(tpos*len(i))
            neg+=tneg
            wn+=int(tneg*len(i))
            neu+=tneu
            comp+=tcomp

    feed_data.append(neg) #author total negative in author conv
    feed_data.append(neu) #author total neutral in author conv 
    feed_data.append(pos) #author total positive in author conv
    feed_data.append(comp) #authortotal compound in author conv
    feed_data.append(wp) #pos word count author
    feed_data.append(wn) #neg word count author


    pcount=0
    for i in messages:
        if i[0]=='p':pcount+=predict_prob([i[5:]])[0]*len(i)

    feed_data.append(pcount) #prof word count author

    final_features = [np.array(feed_data)]
    prediction = model.predict(final_features)
    print(prediction)

    output = round(prediction[0], 2)
    
    if prediction==0:return render_template('index.html', prediction_text=prediction[0])
    else:return render_template('index.html', prediction_text=prediction[0])

   # if output<0.5:return render_template('index.html', prediction_text='He is not suspected as a pedophile')
    #else:return render_template('index.html', prediction_text='He is suspected as a pedophile')
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)







