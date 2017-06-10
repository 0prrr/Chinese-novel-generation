#coding=utf-8

import pickle
import os


def load_text(path):

    '''
    load raw text data
    '''

    input_file = os.path.join(path)

    with open(input_file, 'r') as f:
        text_data = f.read()

    return text_data


def preprocess_and_save_data(text, token_lookup, create_lookup_tables):
    #text = load_text(dataset_path)

    token_dict = token_lookup()
    
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = list(text)
    
    print('preprocess text len:{}'.format(len(text)))

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    
    print('preprocess int_text len:{}'.format(len(int_text)))
    
    
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    return pickle.load(open('params.p', mode='rb'))