from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

class TextNormalization(object):
    def __init__(self):
        super(TextNormalization, self).__init__()

    def stem_sentence(self, sentence):
        porter = PorterStemmer()
        sentence_words = word_tokenize(sentence)
        stem_sentence_words = []
        for word in sentence_words:
            stem_sentence_words.append(porter.stem(word))
        return " ".join(stem_sentence_words)

    def lemma_sentence(self, sentence):
        wordnet_lemmatizer = WordNetLemmatizer()
        sentence_words = word_tokenize(sentence)
        lemma_sentence_words = []
        for word in sentence_words:
            lemma_sentence_words.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        return " ".join(lemma_sentence_words)

