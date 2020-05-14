from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap


# NLP Packages
from textblob import TextBlob,Word
import random
import time

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyse',methods=['POST'])
def analyse():
    start = time.time()
    training = [
        ('Tom Holland is a terrible spiderman.', 'pos'),
        ('a terrible Javert (Russell Crowe) ruined Les Miserables for me...', 'pos'),
        ('The Dark Knight Rises is the greatest superhero movie ever!', 'neg'),
        ('Fantastic Four should have never been made.', 'pos'),
        ('Wes Anderson is my favorite director!', 'neg'),
        ('Captain America 2 is pretty awesome.', 'neg'),
        ('Let\s pretend "Batman and Robin" never happened..', 'pos'),
    ]
    testing = [
        ('Superman was never an interesting character.', 'pos'),
        ('Fantastic Mr Fox is an awesome film!', 'neg'),
        ('Dragonball Evolution is simply terrible!!', 'pos')
    ]
    from textblob import classifiers
    classifier = classifiers.NaiveBayesClassifier(training)
    if request.method == 'POST':
        rawtext = request.form['rawtext'].lower()
        blob = TextBlob(rawtext, classifier=classifier)
        received_text2 = blob
        blob_polarity,blob_subjectivity,blob_sentiment = blob.sentiment.polarity ,blob.sentiment.subjectivity,blob.classify()
        number_of_tokens = len(list(blob.words))
        # Extracting Main Points
        nouns = list()
        for word, tag in blob.tags:
            if tag == 'NN':
                nouns.append(word.lemmatize())
                len_of_words = len(nouns)
                rand_words = random.sample(nouns,len(nouns))
                final_word = set()
                for item in rand_words:
                    word = Word(item).pluralize()
                    final_word.add(word)
                    summary = final_word
                    end = time.time()
                    final_time = end-start


    return render_template('index.html',received_text = received_text2,number_of_tokens=number_of_tokens,blob_polarity=blob_polarity,blob_sentiment=blob_sentiment,blob_subjectivity=blob_subjectivity,summary=summary,final_time=final_time)






if __name__ == '__main__':
    app.run(debug=True)