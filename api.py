#!/usr/bin/python3

from flask import Flask, jsonify, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import urllib.request
import spacy
#import pdfplumber
import os
import re
#from tqdm import tqdm
import logging

from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

import nltk
import nltk.translate.bleu_score as bleu
import nltk.translate.gleu_score as gleu

from rouge import Rouge 


UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'dsgiuhdsfgfigsf'

from werkzeug.routing import Rule
app.url_rule_class = lambda path, **options: Rule('/nlp' + path, **options)

    
################
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        txt = request.form['text']
        summ = request.form['sum']
        gen = request.form['gen']
        doc = nlp(re.sub(r'\n',' ',txt))
        
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
        # excluded sentes
        excl = ""
        for i, s in enumerate(doc.sents):
            for token in s:
                if token.ent_type_ in ['GPE', 'NORP'] or token.pos_ == 'NUM': # or ORG >1 ?
                    #print(token.text, token.ent_type_)
                    excl += s.text
                    print(f'Excluded sentence {i}: {s}')
                    break
        # summarisation similarity
        s1 = nlp(summary)
        s2 = nlp(re.sub(r'\n',' ',summ))
        sum_sim = s1.similarity(s2)
        # generalization similarity
        g1 = nlp(excl)
        g2 = nlp(re.sub(r'\n',' ',gen))
        gen_sim = g1.similarity(g2)
        # BLEU/GLEU summ
        bl1 = summary.split()
        ref = re.sub(r'\n',' ',summ).split()
        # print(bl1,ref)
        bleu_sum = bleu.sentence_bleu([ref], bl1)
        gleu_sum = gleu.sentence_gleu([ref], bl1)
        # BLEU/GLEU generalisation
        bl2 = excl.split()
        ref2 = re.sub(r'\n',' ',gen).split()
        # print(bl2,ref2)
        bleu_gen = bleu.sentence_bleu([ref2], bl2)
        gleu_gen = gleu.sentence_gleu([ref2], bl2)   
        # ROUGE
        rouge = Rouge()
        r_sum = rouge.get_scores(summary, summ)
        r_gen = rouge.get_scores(excl, gen)
                    
        ret = {'summary': summary, 'excluded':excl, 
        'abstract_similarity':sum_sim, 
        'generalization_similarity':gen_sim,
        'BLEU_summ': bleu_sum,
        'GLEU_summ': gleu_sum,
        'BLEU_gen': bleu_gen,
        'GLEU_gen': gleu_gen,
        'ROUGE_sum': r_sum,
        'ROUGE_gen': r_gen}
        
        return jsonify(ret)
                
        
    return '''
    <!doctype html>
    <title>Enter text</title>
    <h1>Enter text for analysis</h1>
    <form method=post>
    Source text:
      <input type=text name=text>
    Expected summarisation:
      <input type=text name=sum>
    Expected generalization (phrases to be removed only):
      <input type=text name=gen>
      <input type=submit value=Send>
    </form>
    '''

if __name__=='__main__':
    nlp = spacy.load("en_core_web_sm")
    app.run(debug=True, port=8075)
