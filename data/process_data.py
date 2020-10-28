import sys
import pandas as pd
import sqlite3

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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


if __name__ == '__main__':
    main()