from abc import abstractmethod
import jieba
import nltk

class Tokenizer():

    pad_token = "<pad>"
    unk_token = "<unk>"
    sos_token = "<sos>"
    eos_token = "<eos>"


    @staticmethod
    @abstractmethod
    def tokenize(sentence) -> list[str]:
        pass


    @classmethod
    def build_vocab(cls, sentence_list, file):

        vocab_set = set()
        for sentence in sentence_list:
            vocab_set.update(cls.tokenize(sentence))

        vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + sorted(list(vocab_set))

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

        self.pad_token_index = self.word2index[self.pad_token]
        self.unk_token_index = self.word2index[self.unk_token]
        self.eos_token_index = self.word2index[self.eos_token]
        self.sos_token_index = self.word2index[self.sos_token]


    def encode(self, sentence):
        return [self.word2index.get(word, self.unk_token_index) for word in self.tokenize(sentence)]


    def decode(self, encoded):
        return [self.index2word.get(index, self.unk_token) for index in encoded]


    def encode_to_seq_len(self, sentence, seq_len):

        out = self.encode(sentence)

        if len(out) < seq_len:
            out = out + [self.pad_token_index] * (seq_len - len(out))
        elif len(out) > seq_len:
            out = out[:seq_len]

        return out
        
