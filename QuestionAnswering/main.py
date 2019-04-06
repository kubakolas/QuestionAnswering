import pickle

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


if __name__== "__main__":
    # train set - zbior pytan i poprawnych odpowiedzi do kazdego pytania
	for data_item in train:
		print('Question:', translate_sent(data_item['question']))
		for answer in data_item['answers']:
			print('Answer: ', get_answer(answer))
		print('=================')