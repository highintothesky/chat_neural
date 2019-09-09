# test the trained model
from process_movie_lines import sentences_to_matrix
from train_attention import infer_nmt
from tensorflow.python.keras.models import load_model
from layers.attention import AttentionLayer
from gensim.models import FastText
import numpy as np
import tensorflow as tf
import layers.attention as k_att

if __name__ == '__main__':
    print('-> Loading att model')
    att_model = load_model('models/att1_full.h5',
                           custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
                                           "AttentionLayer": k_att.AttentionLayer})
    print('-> Loading enc model')
    infer_enc_model = load_model('models/att1_enc.h5',
                           custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
                                           "AttentionLayer": k_att.AttentionLayer})
    print('-> Loading dec model')

    infer_dec_model = load_model('models/att1_dec.h5',
                           custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
                                           "AttentionLayer": k_att.AttentionLayer})
    ft_model = FastText.load('models/fasttext2')
    # infer_enc_model.compile(optimizer='adam', loss='mse')
    # infer_dec_model.compile(optimizer='adam', loss='mse')
    # print(att_model)
    print(att_model.summary())
    sentences = ["sos hey, what's up? eos",
                 "sos kiss my ass eos",
                 "sos Are you really a robot? I can't believe it! eos"]
    mat = sentences_to_matrix(sentences, ft_model, 30)
    print('-> test mat shape:', mat.shape)
    # print(mat)
    for i in range(len(sentences)):
        test_en_seq = np.expand_dims(mat[i,:,:], 0)
        print('-> Test sentence:')
        print(sentences[i])
        print(test_en_seq.shape)
        test_fr, attn_weights = infer_nmt(
            encoder_model=infer_enc_model, decoder_model=infer_dec_model,
            test_en_seq=test_en_seq, emb_len = 100, ft_model = ft_model)
        print('-> Done. output text:')
        print(test_fr)
