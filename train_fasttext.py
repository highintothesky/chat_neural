# gensim model has some aadvantages
from gensim.models import FastText

with open('data/lines_raw.txt') as f:
    content = f.readlines()
content = [x.strip().split() for x in content]

model = FastText(size=100, window=3, min_count=1)

model.build_vocab(sentences=content)
model.train(sentences=content, total_examples=len(content), epochs=20)

print(model.wv['cunt'])
print(model.most_similar(positive=[model.wv['yeah']], topn=10))

model.save('models/fasttext2')
loaded_model = FastText.load('models/fasttext2')
print(loaded_model.most_similar(positive=[loaded_model.wv['yeah']], topn=10))
