import jieba

class MyTokenizer():


    unk_index = 0
    unk_token = "<unk>"


    @staticmethod
    def tokenize(sentence):
        return jieba.lcut(sentence)


    @classmethod
    def build_vocab(cls, sentence_list, file):

        vocab_set = set()
        for sentence in sentence_list:
            vocab_set.update(cls.tokenize(sentence))

        vocab_list = sorted(list(vocab_set))
        vocab_list.insert(cls.unk_index, cls.unk_token)

        with open(file, "wt", encoding="utf-8") as f:
            f.writelines([x + "\n" for x in vocab_list])


    @classmethod
    def from_vocab(cls, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f]
        return cls(vocab_list)


    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}


    def encode(self, sentence):
        return [self.word2index.get(word, self.unk_index) for word in self.tokenize(sentence)]


    def decode(self, encoded):
        return [self.index2word.get(index, self.unk_token) for index in encoded]
