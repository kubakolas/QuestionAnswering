import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def get_pickle(filename):
	return pickle.load(open(filename, 'rb'))

def translate_sent(sent):
	return [vocab[word] for word in sent]

def get_answer(answer_id):
	return translate_sent(answers[answer_id])

vocab = get_pickle('data/vocabulary')   # slowoID - slowo
answers = get_pickle('data/answers')    # odpID - trescOdp{slowoID,...,}
test = get_pickle('data/test1')         # trescPytania{slowoId,...,} - dobraOdpId - {zlaOdpId,...,} ==> LISTA
train = get_pickle('data/train')        # trescPytania{slowoId,...,] - {dobraOdpID,...,}            ==> LISTA

questions = []
for data_item in train:
	question_str = translate_sent(data_item['question'])
	questions.append(question_str)


# lista dokumentów z pytaniami
tagged_questions = [TaggedDocument(words=question, tags=[str(i)]) for i, question in enumerate(questions)]

#tworzymy model doc2vec dla pytan
max_epochs = 2
vec_size = 20
alpha = 0.025
model = Doc2Vec(vector_size=vec_size,
				alpha=alpha,
				min_alpha=0.00025,
				min_count=1,
				dm =1,
                epochs=40)

#budujemy slownik
model.build_vocab(tagged_questions)

#trenujemy model pytaniami
model.train(tagged_questions,
				total_examples=model.corpus_count,
				epochs=model.epochs)

model.save("d2vQuestions.model")
print("Model Saved")

if __name__== "__main__":

    model = Doc2Vec.load("d2vQuestions.model")

    # to find most similar doc using tags
    similar_doc = model.docvecs.most_similar('1')
    print(similar_doc)

    # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
    print(model.docvecs['1'])

	# for data_item in train:
	# 	print('Question:', translate_sent(data_item['question']))
	# 	for answer in data_item['answers']:
	# 		print('Answer: ', get_answer(answer))
	# 	print('=================')