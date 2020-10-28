import sys
import sqlite3
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import re, string
from sklearn.base import TransformerMixin
from joblib import dump, load
from workspace_utils import active_session
import nltk
import ssl
import numpy as np

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

def main():
    """
    NOTE:
    Please run this to install XGBoost
    <conda install -c anaconda py-xgboost>
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
#         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


def load_data(database_filepath):
    engine = sqlite3.connect(database_filepath)
    df = pd.read_sql('select * from message_categories', engine)
    
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        clean_non_ascii = lambda wrd: re.sub(r"[^{}]".format(string.ascii_letters), " ", wrd.lower())
        remove_stop_words = lambda text: ' '.join([w for w in text.split() if not w in stop_words])
        return WordNetLemmatizer().lemmatize(
            remove_stop_words(clean_non_ascii(text))
        )  

    cols = list(df)
    cols.insert(4,'message_cleaned')
    df['message_cleaned'] = df.message.apply(clean_text)
    df = df[cols]
    X, Y = df['message_cleaned'], df[list(df)[5:]]
    return X, Y, list(Y)
    

def tokenize(text):
    pass


def build_model():
    return Pipeline([
  ('tfidf', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
                  encoding='utf-8',
                  input='content', lowercase=True, max_df=1.0, max_features=None,
                  min_df=1, ngram_range=(1, 1), norm='l2', preprocessor=None,
                  smooth_idf=True, stop_words=None, strip_accents=None,
                  sublinear_tf=False, token_pattern='(?u)\\b\\w\\w+\\b',
                  tokenizer=None, use_idf=True, vocabulary=None)),
 ('clf', MultiOutputClassifier(estimator=XGBClassifier(base_score=None, 
                                                       booster=None,
                                                       colsample_bylevel=None,
                                                       colsample_bynode=None,
                                                       colsample_bytree=None, 
                                                       gamma=None,
                                                       gpu_id=None,
                                                       importance_type='gain',
                                                       interaction_constraints=None,
                                                       learning_rate=0.1,
                                                       max_delta_step=None, 
                                                       max_depth=8,
                                                       min_child_weight=None,
                                                       missing=np.nan,
                                                       monotone_constraints=None,
                                                       n_estimators=180, 
                                                       n_jobs=None,
                                                       num_parallel_tree=None,
                                                       objective='binary:logistic',
                                                       random_state=None, 
                                                       reg_alpha=None,
                                                       reg_lambda=None,
                                                       scale_pos_weight=None,
                                                       subsample=None, 
                                                       tree_method=None,
                                                       validate_parameters=None,
                                                       verbosity=None),
                        n_jobs=-1))]
                    )


def evaluate_model(model, X_test, Y_test, category_names):
    predictions = model.predict(X_test)
    def classification_report_csv(col, report):
        """Adapted from:
        https://stackoverflow.com/questions/39662398/scikit-learn-output-
        metrics-classification-report-into-csv-tab-delimited-format
        """
        lines = report.splitlines()
        report, lines = lines[0].split(), lines[2:-4]
        df = pd.DataFrame([ln.split() for ln in lines], columns=['class']+report)
        df['label'] = col
        return df

    report_df = pd.DataFrame()
    yhat_df = pd.DataFrame(predictions, columns=Y_test.columns)
    for col in Y_test.columns:
        report_df = pd.concat([report_df, classification_report_csv(col, 
                                classification_report(Y_test[col], yhat_df[col]))])
    report_df = report_df.reset_index()
    report_df = report_df.drop(['index'], axis=1)
    print(report_df)


def save_model(model, model_filepath):
    dump(model, model_filepath)


if __name__ == '__main__':
    main()