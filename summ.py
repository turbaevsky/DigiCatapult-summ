import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
import re
from heapq import nlargest
### transformers for abstractive summarisation
from transformers import pipeline
### PEGASUS abstractive summarisation
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
### BLEU
import nltk
import nltk.translate.bleu_score as bleu

class Summ:
    """Class to manipulate text performing extractive and abstractive summarisation, 
        extractive generalisation and providing with metrics
    """
    
    def __init__(self, text, exp_sum=None, exp_gen=None):
        """Initialise the object for further anipulation.
        
        Args:
            text (str): the source text to be analysed as a set of strings.
            exp_sum (str): the expected summarisation from the source text, optional.
            exp_gen (str): the expected set of sentences to be removed during extractive generalisation, optional.
        """

        self.text = text
        self.exp_summ = exp_sum
        self.exp_gen = exp_gen
        self.nlp = spacy.load("en_core_web_sm")

        
    def ext_summ(self):
        """ Perform extractive summarisation based on TF-IDF

        Returns:
            set of strings (long string) selected as a summary. Sentences are copied from the source text based on their 'importance' score
        """
        doc = self.nlp(re.sub(r'\n',' ',self.text))
        keyword = []
        stopwords = list(STOP_WORDS)
        pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
        for token in doc:
            if(token.text in stopwords or token.text in punctuation):
                continue
            if(token.pos_ in pos_tag):
                keyword.append(token.text)
        freq_word = Counter(keyword)
        
        max_freq = Counter(keyword).most_common(1)[0][1]
        for word in freq_word.keys():  
            freq_word[word] = (freq_word[word]/max_freq)
        freq_word.most_common(5)
        
        sent_strength={}
        for sent in doc.sents:
            for word in sent:
                if word.text in freq_word.keys():
                    if sent in sent_strength.keys():
                        sent_strength[sent]+=freq_word[word.text]
                    else:
                        sent_strength[sent]=freq_word[word.text]
                        
        summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
        # summary
        final_sentences = [ w.text for w in summarized_sentences ]
        summary = ' '.join(final_sentences)
        return summary

    
    def abs_summ_transf(self, max_length=100):
        """ Perform abstractive summarisation based on transformers 
        
        Args:
            max_lengh (int): maximal length of returned summary, in words, optional.

        Returns:
            set of strings (long string)
        """
        summarizer = pipeline("summarization")
        return summarizer(self.text, max_length=max_length, min_length=5, do_sample=False)[0]['summary_text']

    
    def abs_summ_pegasus(self):
        """ Perform abstractive summarisation based on Google's PEGASUS algorithm 

        Returns:
            set of strings (long string)
        """

        model_name = 'google/pegasus-xsum'
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
        
        batch = tokenizer.prepare_seq2seq_batch([self.text], truncation=True, padding='longest',return_tensors='pt')
        translated = model.generate(**batch)
        return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

    
    def similarity(self, sm=None):
        """ Calculate SpaCy based similarity metric

        Args:
            sm (str): string to be compared with

        Returns:
            float: SpaCy similarity score
        """
        
        if self.exp_summ is not None and sm is not None:
            t1 = self.nlp(self.exp_summ)
            t2 = self.nlp(sm)
            return t1.similarity(t2)
        else:
            return 0

        
    def generalisation(self):
        """ Return sentences to be removed as they have any sensitive information.
        
        The approach to be used is a combination of identification specific part of speech (POS) such as financial numbers, geographical location, name of companies etc. (called  Named Entities Recognition) and linguistic analysis (dependency parsing) to find the specific 'head' words.

        Returns:
            str: set of string to be removed from the source text

        """
        excl = ""
        doc = self.nlp(self.text)
        for i, s in enumerate(doc.sents):
            for token in s:
                #if token.ent_type_ != '' or token.pos_ != '':
                #print(token.text, token.lemma_, token.ent_type_, token.pos_, token.dep_)
                if token.ent_type_ in ['GPE', 'NORP']\
                or token.pos_ == 'NUM'\
                or (token.dep_ == 'ROOT' and token.lemma_ in ['face','compete','include','benefit','evolve',
                                                              'affect','rely','develop','accelerate','invest',
                                                             'acquire']):
                    #print(token.text, token.lemma_, token.ent_type_)
                    excl += (s.text+' ')
                    #print(f'Excluded sentence {i}: {s}')
                    break
        return excl

    
    def root(self):
        """Print root words for each sentence
        """
        
        doc = self.nlp(self.text)
        for i, s in enumerate(doc.sents):
            for token in s:
                if token.dep_ == 'ROOT':
                    print(f'Sentence {i} has {token.lemma_} as a root')

                    
    def bleu_score(self, ex=None):
        """ Returns BLEU score. Corpus score calculation Compares 1 candidate document with multiple 
        sentence and 1+ reference documents also with multiple sentences. Different than averaging BLEU scores of each sentence, it calculates the score by "summing the numerators and denominators 
        for each hypothesis-reference(s) pairs before the division"

        Returns:
           float: BLEU score from zero to 1
        """

        if self.exp_gen is not None and ex is not None:
            return bleu.corpus_bleu([[ex]], [self.exp_gen])
        else:
            return 0
