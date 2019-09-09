# train a word2vec model
# from gensim.models import Word2Vec
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True)
print(model.wv['what'])
print(model.wv['and'])
print(model.wv['?'])
# model2 = gensim.models.Word2Vec(data,min_count = 1, size = 100,
                                             # window = 5, sg = 1)
