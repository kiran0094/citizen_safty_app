from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import numpy as np
model1 = pickle.load(open('email.pickle', 'rb'))
model2 = pickle.load(open('sms.pickle', 'rb'))
model3 = pickle.load(open('url.pickle', 'rb'))
app = Flask(__name__)
def emial(email):
       emails= model1.predit(emails)
       return emails
def url(urls):
     vectorizer = TfidfVectorizer()
     X = vectorizer.fit_transform(urls)
     urlsd=model3.predit.predit(urls)
     return urlsd
def sms(sms):
     vectorizer = TfidfVectorizer()
     vectors = vectorizer.fit_transform(sms['message'])
     smsd= model2.predit(sms)
     return smsd
@app.route('/',methods =["GET", "POST"])
def hello_world():
    if request.method == "POST":
       email = request.form.get("email")
       sms= request.form.get("sms")
       urls = request.form.get("urls")
       emailpredect=emial(email)
       smspredect=sms(sms)
       urlpredect=url(urls)
       print(emailpredect,smspredect,urlpredect)    
    return render_template('index.html')
if __name__ == '__main__':
    app.run()
