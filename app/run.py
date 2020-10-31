import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__, static_url_path='/templates/')


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/etl.db')
df = pd.read_sql_table('message_categories', engine)
label_df = pd.read_sql_table('label_counts', engine).sort_values(by='cnt', ascending=False)
words_df = pd.read_sql_table('word_counts', engine).sort_values(by='cnt', ascending=False)

# load model
model = joblib.load("../models/model.joblib")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker={'color': '#01bcff'}
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_df.cnt.values.tolist(),
                    y=label_df.label.values.tolist(),
                    orientation='h',
					marker={'color': '#32fed2'}
                )
            ],

            'layout': {
                'title': 'Distribution of Labels',
                'yaxis': {
                    'title': "Count",
                    'automargin': True
                },
                'xaxis': {
                    'title': "Label",
                    'automargin': True
                }
            },
        },
        {
            'data': [
                Bar(
                    x=words_df.cnt.values.tolist(),
                    y=words_df.word.values.tolist(),
                    orientation='h',
                    marker={'color': '#dd6cfd'}
                )
            ],

            'layout': {
                'title': 'Distribution of Words',
                'yaxis': {
                    'title': "Count",
                    'automargin': True
                },
                'xaxis': {
                    'title': "Word",
                    'automargin': True
                }
            },
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
