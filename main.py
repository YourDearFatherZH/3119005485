#encoding=utf8

import tensorflow as tf 
import keras 
import numpy as np
import sys
import csv
sys.path.append('../')
from drcn import DRCN
from load_data import load_char_data,load_word_embed,load_char_embed,load_all_data


np.random.seed(1)
tf.set_random_seed(1)

params = {
    'num_classes':2,
    'max_features':1700,
    'embed_size':100,
    'filters':300,
    'kernel_size':3,
    'strides':1,
    'padding':'same',
    'conv_activation_func':'relu',
    'embedding_matrix':[],
    'w_initializer':'random_uniform',
    'b_initializer':'zeros',
    'dropout_rate':0.5,
    'mlp_activation_func':'relu',
    'mlp_num_layers':1,
    'mlp_num_units':256,
    'mlp_num_fan_out':128,
    'task':'Classification',
    'input_shapes':[(48,),(48,),(48,),(48,)],
    'lstm_units':64,
    'num_blocks':1,
    'word_max_features':7300,
    'word_embed_size':100,
}

if __name__ == '__main__':

    org_text_path = sys.argv[1]
    org_change_path = sys.argv[2]
    ans_text_path = sys.argv[3]
    org_text = open(org_text_path, "r",encoding='utf-8')
    org = org_text.read()
    org_change = open(org_change_path, "r",encoding='utf-8')
    add = org_change.read()
    with open('zhanghong.csv', "w",encoding='gbk') as work_path:
        header = ['org', 'org_change', 'similarity']
        writer = csv.DictWriter(work_path, fieldnames=header)
        writer.writeheader()
        writer.writerow({'org': org, 'org_change': add, 'similarity': 1})

    word_embedding_matrix = load_word_embed(params['word_max_features'], params['word_embed_size'])
    char_embedding_matrix = load_char_embed(params['max_features'], params['embed_size'])

    params['embedding_matrix'] = char_embedding_matrix
    params['word_embedding_matrix'] = word_embedding_matrix
    params['embedding_matrix'] = char_embedding_matrix

    backend = DRCN(params)

    p_c_index_test, h_c_index_test, p_w_index_test, h_w_index_test, same_word_test, y_test = load_all_data(
        'zhanghong.csv', maxlen=params['input_shapes'][0][0])
    x_test = [p_c_index_test, h_c_index_test, p_w_index_test, h_w_index_test]
    y_test = keras.utils.to_categorical(y_test, num_classes=params['num_classes'])

    model = backend.build()

    bast_model_filepath = 'best_drcn_model.h5'

    model.load_weights(bast_model_filepath)

    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
        )

    similarity = model.predict(
        x=x_test,
        )

    with open(ans_text_path, "w") as ans_text:
        for tag in similarity:
            ans_text.write(str(tag[1]))