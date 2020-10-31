import sys
import pandas as pd
import sqlite3
import re, string
import nltk, ssl

def check_ntlk_dependencies():
	try:
		_create_unverified_https_context = ssl._create_unverified_context
	except AttributeError:
		pass
	else:
		ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
from nltk.corpus import stopwords

check_ntlk_dependencies()    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving Message and Category data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
		print('Saving Label Count data...\n    DATABASE: {}'.format(database_filepath))
		save_label_counts(df, database_filepath)	
		
		print('Saving Word Count data...\n    DATABASE: {}'.format(database_filepath))
		save_word_frequencies(df, database_filepath)	
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='inner')
    return df


def clean_data(df):
    catdf = df.categories.str.split(';',expand=True)
    orig_cols = set(df)
    for col in list(list(catdf)):
        df[catdf[col].values[0].split('-')[0]] = catdf[col].str.split('-').str[1] 
    del catdf; del df['categories']    
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    conn = sqlite3.connect(database_filename)
    df.to_sql('message_categories', conn, index=False, if_exists='replace')  

	
def save_label_counts(df, database_filepath):
	conn = sqlite3.connect(database_filename)
    label_cols = [col for col in df.columns if col not in ['id', 'message', 'original', 'genre', ]]
    labels = df[label_cols]
    labels = labels.apply(lambda row: [label for label in label_cols if int(row[label]) == 1], axis=1).explode()
    labels = labels.reset_index()
    labels.columns = ['cnt', 'label']
    label_counts = labels.groupby('label').count()
    label_df = label_counts.reset_index().sort_values(by='cnt', ascending=False)
    label_df.to_sql('label_counts', conn, index=False, if_exists='replace')

	
def save_word_frequencies(df, database_filepath, terms = 30): 
	conn = sqlite3.connect(database_filename)
    stop_words = set(stopwords.words('english'))
    clean_non_ascii = lambda wrd: re.sub(r"[^{}]".format(string.ascii_letters), " ", wrd.lower())
    remove_stop_words = lambda text: ' '.join([w for w in text.split() if not w in stop_words])
    df_col = df.message
    df_col = df_col.apply(clean_non_ascii).apply(remove_stop_words)
    all_words = ' '.join([text for text in df_col]) 
    all_words = all_words.split() 
    fdist = nltk.FreqDist(all_words) 
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'cnt':list(fdist.values())}) 

    words_df = words_df.nlargest(columns="cnt", n = terms) 
    words_df.to_sql('word_counts', conn, index=False, if_exists='replace')
    


if __name__ == '__main__':
    main()