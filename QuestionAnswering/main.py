import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import re
#nltk.download('punkt')
stop_words = [","]

def get_pickle(filename):
	return pickle.load(open(filename, 'rb'))

def translate_sent(sent):
	return [vocab[word] for word in sent]

def get_answer(answer_id):
	return translate_sent(answers[answer_id])

vocab = get_pickle('data/vocabulary')   # slowoID - slowo
answers = get_pickle('data/answers')    # odpID - trescOdp{slowoID,...,}
test1 = get_pickle('data/test1')         # trescPytania{slowoId,...,} - {dobraOdpId,...,} - {zlaOdpId,...,} ==> LISTA
train = get_pickle('data/train')        # trescPytania{slowoId,...,] - {dobraOdpID,...,}                   ==> LISTA
test = get_pickle('data/test2')
dev = get_pickle('data/dev')

# questions = []
# for data_item in train:
# 	question_str = translate_sent(data_item['question'])
# 	questions.append(question_str)

#
# # lista dokumentów z pytaniami
# tagged_questions = [TaggedDocument(words=question, tags=[str(i)]) for i, question in enumerate(questions)]
#
# #tworzymy model doc2vec dla pytan
# max_epochs = 2
# vec_size = 20
# alpha = 0.025
# model = Doc2Vec(vector_size=vec_size,
# 				alpha=alpha,
# 				min_alpha=0.00025,
# 				min_count=1,
# 				dm =1,
# 				epochs=40)
#
# #budujemy slownik
# model.build_vocab(tagged_questions)
#
# #trenujemy model pytaniami
# model.train(tagged_questions,
# 				total_examples=model.corpus_count,
# 				epochs=model.epochs)
#
# model.save("d2vQuestions.model")
# print("Model Saved")

if __name__== "__main__":

	# model = Doc2Vec.load("d2vQuestions.model")

	#to find most similar doc using tags
	# similar_doc = model.docvecs.most_similar('1')
	# print(similar_doc)
    #
	# first_di = train[1]
	# print(translate_sent(first_di['question']))
    #
	# first_di = train[10929]
	# print(translate_sent(first_di['question']))
	#to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
	# print(model.docvecs['1'])

	counter = 0
	s = ''
	print('Train set')
	list = []
	text_file = open("trainGoodBad.txt", "w")
	for data_item in test:
		q1 = translate_sent(data_item['question'])
		q_string = "\""
		q_string += " ".join(q1)
		q_string += "\""
		q_string += ","
		for good in data_item['good']:
			ans = get_answer(good)
			ans_string = " ".join(ans)
			ans_string = re.sub(r",", "", ans_string)
			good_s = q_string
			good_s += "\""
			good_s += ans_string
			good_s += "\""
			good_s += ","
			good_s += "\""
			good_s += "1"
			good_s += "\"\n"
			list.append(good_s)
			text_file.write(good_s)
		bad_ctr = 0
		for bad in data_item['bad']:
			ans = get_answer(bad)
			ans_string = " ".join(ans)
			ans_string = re.sub(r",", "", ans_string)
			bad_s = q_string
			bad_s += "\""
			bad_s += ans_string
			bad_s += "\""
			bad_s += ","
			bad_s += "\""
			bad_s += "0"
			bad_s += "\"\n"
			list.append(bad_s)
			text_file.write(bad_s)
			bad_ctr += 1
			if bad_ctr > 5:
				break
		counter += 1
	print(counter)
	text_file.close()


		# print('Question:', q1)
		# for answer in data_item['answers']:
		# 	print('Answer: ', get_answer(answer))
		# print('=================')

	# print('Test set')
	# for data_item in test:
	# 	print('Question:', translate_sent(data_item['question']))
	# 	for answer in data_item['good']:
	# 		print('Good Answer: ', get_answer(answer))
	# 	print('Bad Answer: ', get_answer(data_item['bad'][0]))
	# 	counter += 1
	# 	print('=================')
	# print(counter)