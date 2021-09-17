#encoding=utf8

import tensorflow as tf
import numpy as np
import sys
import csv
sys.path.append('../')
from drcn import DRCN
from load_data import load_word_embed,load_char_embed,load_all_data
from line_profiler_pycharm import profile


np.random.seed(1)
tf.set_random_seed(1)

@profile
def yuce():

    base_params = {
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
        'dropout_rate':0.2,
        'mlp_activation_func':'relu',
        'mlp_num_layers':1,
        'mlp_num_units':256,
        'mlp_num_fan_out':128,
        'input_shapes':[(48,),(48,),(48,),(48,)],
        'task':'Classification',
        'lstm_units':64,
        'num_blocks':1,
        'word_max_features':7300,
        'word_embed_size':100
    }

    org_text_path = sys.argv[1]
    org_change_path = sys.argv[2]
    ans_text_path = sys.argv[3]
    org_text = open(org_text_path, "r", encoding='utf-8')
    org = org_text.read()
    org_change = open(org_change_path, "r", encoding='utf-8')
    change = org_change.read()
    with open('zhanghong.csv', "w", encoding='gbk') as work_path:
        header = ['org', 'org_change', 'similarity']
        writer = csv.DictWriter(work_path, fieldnames=header)
        writer.writeheader()
        writer.writerow({'org': org, 'org_change': change, 'similarity': 1})

    word_embedding_matrix = load_word_embed(base_params['word_max_features'], base_params['word_embed_size'])
    char_embedding_matrix = load_char_embed(base_params['max_features'], base_params['embed_size'])

    base_params['embedding_matrix'] = char_embedding_matrix
    base_params['word_embedding_matrix'] = word_embedding_matrix

    backend = DRCN(base_params)

    p_c_index_test, h_c_index_test, p_w_index_test, h_w_index_test, same_word_test, _ = load_all_data(
        'zhanghong.csv', maxlen=base_params['input_shapes'][0][0])
    x_test = [p_c_index_test, h_c_index_test, p_w_index_test, h_w_index_test]

    model = backend.build()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    bast_model_filepath = 'best_drcn_model.h5'

    model.load_weights(bast_model_filepath)

    end = model.predict(
        x=x_test,
    )

    print(end[0][1])

    with open(ans_text_path, "w") as ans_text:
        if len(change) == 0:
            ans_text.write(str(0))
        else:
            ans_text.write(str(end[0][1]))

if __name__ == '__main__':

    yuce()