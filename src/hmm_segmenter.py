#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:Name:
    hmm_segmenter.py

:Authors:
    Florian Boudin (florian.boudin@univ-nantes.fr)
    Coraline Marie

:Date:
    22 july 2013 (creation)

:Description:
    A HMM Japanese segmenter.
"""

import re
import sys
import xml.sax
import codecs
import bisect


################################################################################
# Main function handling function calls
################################################################################
def main(argv):

    if len(argv) != 3:
        print "Usage : hmm_segmenter.py training_file test_file output_file"
        sys.exit()



    # katakana_lst =
    # punctuation_lst =
    # other_lst =

    train_set = content_handler(argv[0])
    test_set = content_handler(argv[1])

    model = train_total(train_set.sentences)
    #model_bigram = train_bigram(train_set.sentences)
    #model_trigram = train_trigram(train_set.sentences)

    handle = codecs.open(argv[2], 'w', 'utf-8')
    handle.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
    handle.write('<dataset>\n')
    for i in range(len(test_set.raw_sentences)):
        segmented_sentence = word_segmentation(model, test_set.raw_sentences[i])
        #segmented_sentence = word_segmentation(model_bigram, model_trigram, test_set.raw_sentences[i])
        handle.write('\t<sentence sid="'+str(i)+'">\n')
        handle.write('\t\t<raw>'+segmented_sentence+'</raw>\n')
        handle.write('\t</sentence>\n')
    handle.write('</dataset>')
    handle.close()

################################################################################


################################################################################
# Useful functions
################################################################################
def add_one(dic, value):
    if not dic.has_key(value):
        dic[value] = 0.0
    dic[value] += 1.0


def get_alphabet(caracter):
    if 0x3040 <= ord(caracter) <= 0x309F:
        return 'hiragana'
    elif 0x30A0 <= ord(caracter) <= 0x30FF:
        return 'katakana'
    elif 0xFF01 <= ord(caracter) <= 0xFFEF:
        return 'romaji'
    elif 0x300 <= ord(caracter) <= 0x303F:
        return 'other'
    else:
        return 'kanji'

################################################################################


################################################################################
# XML Sax Parser for KNBC xml file
################################################################################
class content_handler(xml.sax.ContentHandler):
    #-T-----------------------------------------------------------------------T-
    def __init__(self, path):
        # Tree for pushing/poping tags, attrs
        self.tree = []
        
        # Buffer for xml element
        self.buffer = ''

        # Container for the sentence list
        self.sentences = []
        self.current_sentence = []
        self.raw_sentences = []

        # Construct and launch the parser
        parser = xml.sax.make_parser()
        parser.setContentHandler(self)
        parser.parse(path)
    #-B-----------------------------------------------------------------------B-
        
    #-T-----------------------------------------------------------------------T-
    def startElement(self, name, attrs):
        self.tree.append((name, attrs))
    #-B-----------------------------------------------------------------------B-
        
    #-T-----------------------------------------------------------------------T-
    def characters(self, data):
        self.buffer += data.strip()
    #-B-----------------------------------------------------------------------B-
        
    #-T-----------------------------------------------------------------------T-
    def endElement(self, name):
        tag, attrs = self.tree.pop()

        if tag == 'token':
            self.current_sentence.append(self.buffer)

        elif tag == 'sentence':
            self.sentences.append(self.current_sentence)
            self.current_sentence = []

        elif tag == 'raw':
            self.raw_sentences.append(self.buffer)

        # Flush the buffer
        self.buffer = ''
    #-B-----------------------------------------------------------------------B-
################################################################################



# ################################################################################
# # Training function bigram (base)
# ################################################################################
# def train_bigram(sentences):   # def train(sentences):
#
#     observation_prob = {}
#     state_trans_prob = {}
#     alphabet_trans_prob = {}
#     alphabet_type = ["hiragana", "katakana", "romanji", "kanji", "other"]
#
#     #Initialisation alphabet probability
#     for element in alphabet_type:
#         alphabet_trans_prob[element] = {}
#         for element2 in alphabet_type:
#             alphabet_trans_prob[element][element2] = {'c': 0.0, 'b': 0.0}
#
#     # Iterates through the sentences
#     for sentence in sentences:
#
#         for i in range(len(sentence)):
#             sentence[i] = 'c'.join(sentence[i])		# rajoute un c entre chaque caractère non coupé
#         annotated_sequence = 'b'.join(sentence)		# rajoute un b à la fin de chaque "mot"
#
#         previous_state = ''
#         current_state = ''
#
#         for i in range(0, len(annotated_sequence) - 1, 2):
#             observation = annotated_sequence[i:i + 3]
#             bigram = observation[0] + observation[2]
#
#             current_state = observation[1]
#
#             alphabet_trans_prob[get_alphabet(observation[0])][get_alphabet(observation[2])][current_state] += 1
#
#             if not observation_prob.has_key(bigram):
#                 observation_prob[bigram] = {'c': 0.0, 'b': 0.0}
#             observation_prob[bigram][current_state] += 1.0
#
#             if previous_state != '':
#                 add_one(state_trans_prob, (previous_state, current_state))
#             previous_state = current_state
#
#     # Normalize observation probabilities
#     for bigram in observation_prob:
#         norm_factor = observation_prob[bigram]['c'] + observation_prob[bigram]['b']
#         observation_prob[bigram]['c'] /= norm_factor
#         observation_prob[bigram]['b'] /= norm_factor
#
#     # Normalize transition probabilities
#     norm_factor = 0.0
#     for t in state_trans_prob:
#         norm_factor += state_trans_prob[t]
#     for t in state_trans_prob:
#         state_trans_prob[t] /= norm_factor
#
#
#     return [observation_prob, state_trans_prob, alphabet_trans_prob]
# ################################################################################
#
#
#
# ################################################################################
# # Training function trigram + gestion de l'alphabet
# ################################################################################
# def train_trigram(sentences):
#
#     observation_prob = {}
#     state_trans_prob = {}
#     alphabet_trans_prob = {}
#
#     alphabet_type = ["hiragana", "katakana", "romanji", "kanji", "other"]
#     for element in alphabet_type:
#         alphabet_trans_prob[element] = {}
#         for element2 in alphabet_type:
#             alphabet_trans_prob[element][element2] = {'c': 0.0, 'b': 0.0}
#
#     # Iterates through the sentences
#     for sentence in sentences:
#         annotated_sequence = 'b'.join(sentence)
#         current_state = ''
#         previous_state = ''
#         previous_second_state = ''
#
#         for i in range(0, len(annotated_sequence) - 3, 2):
#             observation = annotated_sequence[i:i + 5]
#             trigram = observation[0] + observation[2] + observation[4]
#             current_state = observation[1]
#
#             alphabet_trans_prob[get_alphabet(observation[0])][get_alphabet(observation[2])][current_state] += 1
#
#             if not observation_prob.has_key(trigram):
#                 observation_prob[trigram] = {'c': 0.0, 'b': 0.0}
#             observation_prob[trigram][current_state] += 1.0
#
#             if previous_second_state != '':
#                 add_one(state_trans_prob, (previous_second_state, previous_state, current_state))
#             elif previous_state != '':
#                 add_one(state_trans_prob, (previous_state, current_state))
#             previous_second_state = previous_state
#             previous_state = current_state
#
#     # Normalize observation probabilities
#     for trigram in observation_prob:
#         norm_factor = observation_prob[trigram]['c'] + observation_prob[trigram]['b']
#         observation_prob[trigram]['c'] /= norm_factor
#         observation_prob[trigram]['b'] /= norm_factor
#
#     # Normalize transition probabilities
#     norm_factor = 0.0
#     for t in state_trans_prob:
#         norm_factor += state_trans_prob[t]
#     for t in state_trans_prob:
#         state_trans_prob[t] /= norm_factor
#
#
#     return [observation_prob, state_trans_prob, alphabet_trans_prob]
# ################################################################################



################################################################################
# Training function bigrams + trigrams backoff + alphabets
################################################################################
def train_total(sentences):

    observation_prob = {}
    state_trans_prob = {}
    alphabet_prob_bigram = {}
    # alphabet_prob_trigram = {}

    var_b = 0
    var_c = 0


    # init alphabet probability
    alphabet_lst = ["hiragana", "katakana", "romaji", "kanji", "other"]
    for element in alphabet_lst:
        alphabet_prob_bigram[element] = {}
        for element2 in alphabet_lst:
            alphabet_prob_bigram[element][element2] = {'c': 0.0, 'b': 0.0}

    # for element in alphabet_lst:
    #     for element2 in alphabet_lst:
    #         alphabet_prob_trigram[(element,element2)] = {}
    #         for element3 in alphabet_lst:
    #             alphabet_prob_trigram[(element, element2)][element3] = {'c': 0.0, 'b': 0.0}



    # Iterates through the sentences

    for sentence in sentences:
        for i in range(len(sentence)):
            sentence[i] = 'c'.join(sentence[i])
        annotated_sequence = 'b'.join(sentence)

        current_state = ''
        previous_state = ''
        previous_second_state = ''


        for i in range(0, len(annotated_sequence)-3, 2):
            observation = annotated_sequence[i:i+5]
            bigram = observation[0] + observation[2]
            trigram = observation[0] + observation[2] + observation[4]
            current_state = observation[1]
            alphabet_prob_bigram[get_alphabet(observation[0])][get_alphabet(observation[2])][current_state] += 1
            # alphabet_prob_trigram[(get_alphabet(observation[0]),get_alphabet(observation[2]))][get_alphabet(observation[4])][current_state] += 1

            if not observation_prob.has_key(trigram):
                observation_prob[trigram] = {'c': 0.0, 'b': 0.0}
            observation_prob[trigram][current_state] += 1.0
            if not observation_prob.has_key(bigram):
                observation_prob[bigram] = {'c': 0.0, 'b': 0.0}
            observation_prob[bigram][current_state] += 1.0
            if current_state == 'b':
                var_b += 1
            else:
                var_c += 1


            if previous_second_state != '':
                add_one(state_trans_prob, (previous_second_state, previous_state, current_state))
            elif previous_state != '':
                add_one(state_trans_prob, (previous_state, current_state))
            previous_second_state = previous_state
            previous_state = current_state

    # Normalize observation probabilities
    for element in observation_prob:
        norm_factor = observation_prob[element]['c'] + observation_prob[element]['b']
        observation_prob[element]['c'] /= norm_factor
        observation_prob[element]['b'] /= norm_factor

    # Normalize transition probabilities
    norm_factor = 0.0
    for t in state_trans_prob:
        norm_factor += state_trans_prob[t]
    for t in state_trans_prob:
        state_trans_prob[t] /= norm_factor

    # norm_factor = 0.0
    # for i in alphabet_prob_bigram.keys():
    #     for j in alphabet_prob_bigram[i].keys():
    #         for k in alphabet_prob_bigram[i][j].keys():
    #             norm_factor += alphabet_prob_bigram[i][j][k]
    # for i in alphabet_prob_bigram.keys():
    #     for j in alphabet_prob_bigram[i].keys():
    #         for k in alphabet_prob_bigram[i][j].keys():
    #             alphabet_prob_bigram[i][j][k] /= norm_factor
    #print ("b = " + str(var_b) + " c = " + str(var_c))
    return [observation_prob, state_trans_prob, alphabet_prob_bigram]
################################################################################


################################################################################
# Function for word segmentation
################################################################################
def word_segmentation(model, sentence):

    observation_prob = model[0]
    state_trans_prob = model[1]
    alphabet_prob = model[2]

    unseen_prob_b = 0.02
    unseen_prob_c = 0.01

    observations = []
    lattice = []


    for i in range(len(sentence)-1):
        bigram = sentence[i:i+2]
        trigram = sentence[i:i+3]

        observations.append(trigram)

        if observation_prob.has_key(trigram):
            lattice.append(observation_prob[trigram])
            lattice[-1]['c'] = max(unseen_prob_c, lattice[-1]['c'])
            lattice[-1]['b'] = max(unseen_prob_b, lattice[-1]['b'])
        elif observation_prob.has_key(bigram):
            lattice.append(observation_prob[bigram])
            lattice[-1]['c'] = max(unseen_prob_c, lattice[-1]['c'])
            lattice[-1]['b'] = max(unseen_prob_b, lattice[-1]['b'])
        else:
            lattice.append(alphabet_prob[get_alphabet(bigram[0])][get_alphabet(bigram[1])])
            lattice[-1]['c'] = max(unseen_prob_c, lattice[-1]['c'])
            lattice[-1]['b'] = max(unseen_prob_b, lattice[-1]['b'])


    # Use viterbi to decode the lattice

    # Initialisation
    V = []
    bisect.insort(V, (lattice[0]['c'], ['c']))
    bisect.insort(V, (lattice[0]['b'], ['b']))

    # Looping the observations
    for i in range(1, len(observations)):

        # Select the n-best paths
        best_V = []
        best_V.append(V[-1])
        highest_prob = best_V[0][0]
        for j in range(len(V)-2,-1,-1):
            if V[j][0] >= highest_prob:
                best_V.append(V[j])
            else:
                break

        new_V = []
        for prob, stack in best_V:
            # Compute probability for continuation
            prob_c = prob * lattice[i]['c'] * state_trans_prob[(stack[-1], 'c')]
            bisect.insort(new_V, (prob_c, stack+['c']))
            prob_b = prob * lattice[i]['b'] * state_trans_prob[(stack[-1], 'b')]
            bisect.insort(new_V, (prob_b, stack+['b']))

        V = list(new_V)

    best_V = V[-1][1]

    # Sentence segmentation
    segmented_sentence = []
    for i in range(len(sentence)-1):
        if best_V[i] == 'c':
            segmented_sentence.append(sentence[i])
        else:
            segmented_sentence.append(sentence[i]+" ")
    segmented_sentence.append(sentence[-1])
    return ''.join(segmented_sentence)


################################################################################

################################################################################
# To launch the script
################################################################################
if __name__ == "__main__":
    main(sys.argv[1:])
################################################################################
