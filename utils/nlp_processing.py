# Installing the required packages for the code to run.
# pip install underthesea==1.3.5a3
# pip install underthesea[deep]
# pip install pyvi
# pip install translate
# pip install googletrans==3.1.0a0

# Importing the libraries that we will use in this notebook.
import googletrans
import translate
import underthesea
from pyvi import ViUtils, ViTokenizer
from difflib import SequenceMatcher

class Translation():
    def __init__(self, from_lang='vi', to_lang='en', mode='google'):
        # The class Translation is a wrapper for the two translation libraries, googletrans and translate. 
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode in 'googletrans':
            self.translator = googletrans.Translator()
        elif mode in 'translate':
            self.translator = translate.Translator(from_lang=from_lang,to_lang=to_lang)

    def preprocessing(self, text):
        """
        It takes a string as input, and returns a string with all the letters in lowercase

        :param text: The text to be processed
        :return: The text is being returned in lowercase.
        """
        return text.lower()

    def __call__(self, text):
        """
        The function takes in a text and preprocesses it before translation

        :param text: The text to be translated
        :return: The translated text.
        """
        text = self.preprocessing(text)
        return self.translator.translate(text) if self.__mode in 'translate' \
                else self.translator.translate(text, dest=self.__to_lang).text

class Text_Preprocessing():
    def __init__(self, stopwords_path='./dict/vietnamese-stopwords-dash.txt'):
        with open(stopwords_path, 'rb') as f:
            lines = f.readlines()
        self.stop_words = [line.decode('utf8').replace('\n','') for line in lines]

    def find_substring(self, string1, string2):
        """
        It uses the SequenceMatcher class from the difflib module to find the longest matching substring
        between two strings
        
        :param string1: The first string to be compared
        :param string2: The string to search for
        :return: The longest common substring between string1 and string2.
        """
        match = SequenceMatcher(None, string1, string2, autojunk=False).find_longest_match(0, len(string1), 0, len(string2))
        return string1[match.a:match.a + match.size].strip()

    def remove_stopwords(self, text):
        """
        - Tokenize the text
        - Remove stopwords
        - Return the text
        
        :param text: The text to be cleaned
        :return: A string of words that are not in the stopwords list.
        """
        text = ViTokenizer.tokenize(text)
        return " ".join([w for w in text.split() if w not in self.stop_words])

    def lowercasing(self, text):
        return text.lower() 

    def uppercasing(self, text):
        return text.upper()

    def add_accents(self, text): 
        """
        It takes a string, and returns a string with all the Vietnamese accents added
        
        :param text: the text to be converted
        :return: The return value is a string.
        """
        return ViUtils.add_accents(u"{}".format(text))

    def remove_accents(self, text): 
        """
        It removes accents from Vietnamese text
        
        :param text: The text to be processed
        """
        return ViUtils.remove_accents(u"{}".format(text))

    def sentence_segment(self, text):
        """
        It takes a string of text as input and returns a list of strings, where each string is a
        sentence
        
        :param text: the text to be segmented
        :return: A list of sentences
        """
        return underthesea.sent_tokenize(text)

    def text_norm(self, text):
        """
        It takes a string as input, and returns a string as output
        
        :param text: the text to be normalized
        """
        return underthesea.text_normalize(text)  

    def text_classify(self, text):
        """
        The function takes in a string of text, and returns a list of tuples, where each tuple contains
        a label and a score
        
        :param text: The text to be classified
        :return: A list of tuples.
        """
        return underthesea.classify(text)

    def sentiment_analysis(self, text):
        """
        The function takes in a string of text, and returns a dictionary with the keys 'neg', 'neu',
        'pos', and 'compound'
        
        :param text: The text to be analyzed
        :return: A dictionary with the following keys:
        """
        return underthesea.sentiment(text)

    def __call__(self, text):
        """
        It takes a string of text, lowercases it, adds accents, normalizes it, and then classifies it
        
        :param text: the text to be classified
        :return: The categories of the text.
        """
        text = self.lowercasing(text)
        text = self.remove_stopwords(text)
        # text = self.remove_accents(text)
        # text = self.add_accents(text)
        text = self.text_norm(text)
        categories = self.text_classify(text)
        return categories

class Tagging():
    def __init__(self):
        pass

    def word_segment(self, text, format=None):
        """
        The function takes a string as input, and returns a list of words
        
        :param text: The text to be segmented
        :param format: str, optional (default=None)
        :return: A list of words
        """
        return underthesea.word_tokenize(text, format=format)

    def POS_tagging(self, text):
        """
        It takes a text as input, and returns a list of tuples, where each tuple contains a word and its
        corresponding part of speech
        
        :param text: the text to be processed
        :return: A list of tuples.
        """
        return underthesea.pos_tag(text)

    def chunking(self, text):
        """
        The function takes a text as an input, then it uses the underthesea library to chunk the text
        and return the result
        
        :param text: the text to be chunked
        :return: A list of tuples.
        """
        return underthesea.chunk(text)

    def dependency_parsing(self, text):
        """
        The function takes a text as input, and returns a list of tuples, each tuple contains the word,
        its part of speech, and its dependency
        
        :param text: the text to be analyzed
        :return: A list of tuples. Each tuple contains the word, its part of speech, and its dependency.
        """
        return underthesea.dependency_parse(text)

    def named_entity_recognition(self, text):
        """
        The function takes a string as input and returns a list of tuples. Each tuple contains the word
        and its corresponding entity
        
        :param text: The text to be analyzed
        :return: A list of tuples.
        """
        return underthesea.ner(text)