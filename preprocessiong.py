import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers


def custom_standardization(input_data):
    return tf.strings.lower(input_data)


def vectorize_text(text, max_features=5000, sequence_length=500):

    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    vectorize_layer.adapt(text)
    return vectorize_layer(text)


def get_url_freq(url):
    d = dict(pd.DataFrame(url).groupby('question_user_page').apply(len))
    return url.apply(lambda x: d[x])


def preprocess_question_title(title_text):
    return vectorize_text(title_text, 2000, 50)


def preprocess_question_body(title_body):
    return vectorize_text(title_body)


def preprocess_answer(answer):
    return vectorize_text(answer)


def preprocess_question_user_page(user_page_url):
    return get_url_freq(user_page_url)


def preprocess_answer_user_page(user_page_url):
    return get_url_freq(user_page_url)


def preprocess_site_page(site_page_url):
    return get_url_freq(site_page_url)


def preprocess_category(user_page_url):
    return get_url_freq(user_page_url)
