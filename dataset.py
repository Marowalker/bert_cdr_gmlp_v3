import random

import numpy as np
from nltk.corpus import wordnet as wn
import constants
from sklearn.utils import shuffle
import tensorflow as tf
from keras.utils import pad_sequences
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
import re

np.random.seed(13)

STOP_WORD = {
    'been', 'than', 'can', 'selves', 'index', 'becomes', 'date', 'inner', 'hereby', 'looks', 'that', 'non', 't',
    'giving', 'unlikely', 'help', 'definitely', 'do', 'latter', 'at ', "haven't", 'namely', 'fix', 'therein', 'eight',
    'hi', 'along', 'seriously', 'keep    keeps', 'beginnings', 'also', 'welcome', 'importance', 'nonetheless', 'vs',
    'un', 'went', 'until', 'truly', 'whos', 'take', 'liked', 'am', 'nothing', 'now', 'anyone', 'all', 'sup', 'inward',
    'currently', 'an ', 'where', 'noted', 'f', 'mainly', 'next', 'given', 'mostly', 'act', "'s", 'somewhere', "she'll",
    'allow', 'got', 'anyhow', 'begin', 'wed', 'it', "wasn't", 'et', 'everyone', "'ve", 'oh', 'viz', 'on', 'found',
    "t's", 'thousand', "didn't", 'forth', 'despite', 'accordingly', 'anyway', 'as', 'is', "doesn't", 'results', 'at',
    'example', 'necessary', 'particular', 'n', "who's", "he's", 'beginning', 'suggest', 'most', 'affecting', 'goes',
    'followed', 'was ', 'anything', 've', 'elsewhere', 'nowhere', 'not', 'sensible', 'h', 'pp', 'o', 'itself', 'okay',
    'w', 'seem', 'obviously', 'whom', 'there', 'immediate', 'g', "that've", 'ran', 'recent', 'mg', 'last', 'above',
    'ending', 'nine', 'ask', 'important', 'whats', 'therere', 'whence', 'adj', 'werent', "ain't", 'away', 'over',
    'presumably', 'whod', 'hid', 'already', 'please', 'third', 'hed', 'moreover', 'tried', 'gone', 'ignored', 'who ',
    'better', 'wherein', 'consider', 'keeps', 'first', 'howbeit', 'throughout', 'associated', 'herein', 'give', 'y',
    'serious', 'else', 'sha', 'other', 'perhaps', 'everything', 'rather', "you've", 'ought', 'look', 'doing', 'knows',
    'exactly', 'have', 'ai', 'on ', 'causes', 'before', 'right', 'want', 'somehow', 'ever', 'her', "he'd", 'or ', 'ltd',
    'be ', 'of', 'within', "wouldn't", 'merely', 'ourselves', 'r', "weren't", 'just', 'allows', 'twice', 'three',
    "here's", 'aside', 'they', 'what', 'specifically', 'you', 'refs', 'when', 'later', 'since', 'recently', 's',
    'means', 'whatever', 'became', 'self', 'normally', "there's", 'mr', 'mean', 'almost', 'had', 'only', 'very',
    "hadn't", "c's", 'them', 'toward', 'thats', 'successfully', 'beside', 'under', 'shes', 'around', 'try', 'why',
    'came', 'section', 'respectively', 'gotten', 'same', 'six', 'below', 'j', 'appreciate', 'c', 'nevertheless', "a's",
    'ok', 'whose', 'well', 'pages', 'far', 'cannot', 'nos', 'changes', 'likely', 'may', 'consequently', "it'd",
    'reasonably', 'million', 'someone', 'onto', 'theyre', 'whenever', 'he', 'if', 'were', 'run', 'appropriate', 'say',
    "shouldn't", 'regards', 'miss', 'clearly', 'heres', 'myself', 'ts', 'substantially', 'thereto', 'let', 'et-al',
    'wheres', 'thereafter', 'after', 'per', 'provides', 'each', 'keep', 'anyways', "why's", 'or', 'briefly', "how's",
    'get', "'m", "it'll", 'promptly', 'once', 'seems', 'wonder', 'uses', 'ff', 'various', "c'mon", 'nobody', 'na',
    'again', 'indeed', 'therefore', 'saying', 'alone', 'two', 'm', 'but', 'l', 'will ', "what's", 'past', "there'll",
    "i'd", 'described', 'ed', 'p', 'to', 'a ', 'shown', 'make', 'in', 'many', 'immediately', 'wherever', 'the ', 'sub',
    'yourself', 'though', 'everywhere', 'tries', 'yet', 'corresponding', 'besides', 'showns', 'about', 'either',
    'himself', 'much', 'slightly', 'biol', 'such', 'will', 'primarily', 'nt', "that'll", 'through', 'nay', 'relatively',
    'end', 'lest', 'nd', 'indicates', 'become', 'then', 'ca', 'itd', 'possibly', 'que', 'hence', 'downwards',
    'together', 'via', 'inasmuch', 'im', 'believe', 'further', 'usefully', 'an', 'specifying', "they're", 'little',
    'etc', 'edu', 'did', 'are', 'us', 'how', 'their', 'following', 'during', 'own', 'using', 'effect', 'cant', 'the',
    "they'll", "we're", 'know', 'related', 'added', 'yourselves', 'whereas', 'somebody', 'apparently', 'hello',
    'something', 'seven', "n't", 'unto', 'due', 'accordance', 'thus', 'lets', 'anybody', "there've", 'rd', 'although',
    'present', 'containing', 'placed', 'so', 'still', 'thereupon', 'ref', 'really', 'thence', 'meanwhile', 'taken',
    'whomever', 'co', 'does', 'herself', 'second', 'upon', 'auth', 'show', "it's", 'comes', 'considering', 'thorough',
    "isn't", 'ups', 'beforehand', 'behind', 'hither', "hasn't", 'towards', 'affected', 'significantly', 'him', 'qv',
    'hundred', "she'd", 'my', 'wo', 'these', 'announce', 'while', 'says', 'arent', 'nearly', 'cause', 'never', 'ours'
    'by', "she's", 'arise', 'couldnt', 'because', 'contain', 'need', "aren't", 'off', 'certainly', 'has', "who'll",
    'shows', 'should', 'specified', 'thanks', 'new', 'me', 'thereby', 'approximately', 'about ', 'seeing', 'and', 're',
    'hereafter', 'up', 'gives', 'wants', 'whereafter', 'information', 'shed', 'near', 'part', "we'll", 'several',
    'line', "where's", 'wasnt', "you're", "don't", 'thoroughly', 'different', 'way', 'formerly', 'seen', 'plus', 'able',
    'she', 'I ', 'noone', 'as ', 'needs', 'a', 'asking', 'sometime', 'sent', 'stop', 'com ', 'too', 'sorry', 'seeming',
    'obtained', 'ones', 'going', "'d", 'tip', "i'll", 'according', 'latterly', 'your', 'could', 'greetings', 'for ',
    'to ', 'theyd', 'particularly', 'out', "we've", 'showed', 'whim', 'regardless', 'value', 'inc', 'furthermore', 'ie',
    'hereupon', 'thanx', 'whither', 'without', 'predominantly', 'significant', 'for', 'youd', 'vol', 'said', 'e',
    'come', 'specify', 'somethan', 'thru', 'across', 'd', 'hes', 'mrs', 'be', 'any', 'whole', 'theres', 'between',
    'five', "won't", 'lately', "when's", 'was', 'al', 'more', 'ah', 'wouldnt', 'otherwise', 'no', "shan't", 'apart',
    'beyond', 'enough', 'former', 'affects', 'ninety', "that's", 'might', 'proud', 'least', 'available', 'actually',
    'used', 'home', 'would', "he'll", 'certain', 'id', "you'd", 'thou', 'makes', 'similarly', 'themselves', 'put', 'z',
    'yours', 'kg', 'however', 'sec', 'of ', "i'm", 'everybody', 'outside', 'anywhere', 'both', 'every', 'possible',
    'whereupon', 'its', 'others', 'trying', 'indicate', 'somewhat', 'similar', 'brief', 'awfully', 'who', 'looking',
    'follows', 'use', 'thereof', 'kept', 'happens', 'km', 'except', 'research', 'necessarily', 'v', 'we', 'resulted',
    'always', 'seemed', "can't", 'concerning', 'is ', 'thered', 'com', 'wish', 'made', 'meantime', 'whether', 'throug',
    'none', 'resulting', 'youre', 'shall', 'potentially', "what'll", 'probably', 'with', 'maybe', 'zero', 'regarding',
    'even', 'vols', 'quite', 'ml', 'words', 'back', 'q', 'quickly', 'k', 'among', 'another', 'which', 'tends',
    'amongst', 'usually', 'known', 'into', 'overall', "'ll", 'eighty', 'what ', 'aren', "i've", 'it ', 'often', 'sure',
    'th', 'anymore', 'thank', 'begins', 'omitted', 'ord', 'unlike', "'", 'ex', 'afterwards', 'b', 'taking', 'against',
    'eg', 'contains', 'strongly', 'world', 'soon', 'old', 'wont', 'gets', 'whereby', 'insofar', 'done', 'usefulness',
    'one', "let's", "they'd", 'by ', 'entirely', 'theirs', 'this', 'course', 'unfortunately', 'gave', 'less', 'must',
    'abst', 'invention', 'neither', "we'd", 'sometimes', 'being', 'i', "they've", 'novel', 'his', 'hers', 'think',
    'fifth', 'tell', 'especially', 'nor', 'x', 'appear', 'readily', 'four', 'mug', 'largely', 'becoming', 'getting',
    'instead', 'obtain', 'willing', 'sufficiently', 'widely', 'few', 'yes', 'www', 'name', 'previously', "you'll",
    'owing', "'re", 'our', 'see', 'page', 'secondly', 'hopefully', "couldn't", 'some', "mustn't", 'u', 'down', 'from',
    'best', 'hardly', 'useful', 'poorly', 'like', 'indicated', 'thoughh', 'having', 'unless', 'whoever', 'til', '-',
    'go', 'took', 'in ', 'those', 'here', 'saw', 'are ', 'I'
}


def clean_lines(lines):
    cleaned_lines = []
    for line in lines:
        l = line.strip().split()
        if len(l) == 1:
            cleaned_lines.append(line)
        else:
            pair = l[0]
            if '-1' not in pair:
                cleaned_lines.append(line)
    return cleaned_lines


def get_wordnet_pos(treebank_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        # As default pos in lemmatization is Noun
        return wn.NOUN


def hypernym(word, pos):
    """

    :params word: str:
    :return:
    """
    synonyms = wn.synsets(word, get_wordnet_pos(pos))
    hypernyms = []
    for syn in synonyms:
        hyper = syn.hypernyms()
        hypernyms += hyper
    lemmas = [h.lemmas()[i].name() for h in hypernyms for i in range(len(h.lemmas()))]

    return lemmas


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def my_pad_sequences(sequences, pad_tok, max_sent_length, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        max_sent_length: the maximum length of the padded sentence
        dtype: the type of the final return value
        nlevels: the level (no. of dimensions) of the padded matrix
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        # max_length = max(map(lambda x: len(x), sequences))
        max_length = max_sent_length
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))

        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_sent_length)

    return np.array(sequence_padded), sequence_length


def parse_words(raw_data, cid_only=False):
    raw_data = clean_lines(raw_data)
    all_words = []
    all_poses = []
    all_synsets = []
    all_relations = []
    all_positions = []
    all_labels = []
    all_identities = []
    all_triples = []
    pmid = ''
    all_lens = []
    doc_len = 0
    for line in raw_data:
        l = line.strip().split()
        if len(l) == 1:
            pmid = l[0]
            # doc_len = l[1]
        else:
            pair = l[0]
            label = l[1]
            if (label == 'CID' and cid_only) or (not cid_only):
                chem, dis = pair.split('_')
                all_triples.append([chem, dis])

                joint_sdp = ' '.join(l[2:])
                sdps = joint_sdp.split("-PUNC-")
                for sdp in sdps:
                    # S xuoi
                    nodes = sdp.split()
                    words = []
                    poses = []
                    synsets = []
                    positions = []
                    relations = []
                    for idx, node in enumerate(nodes):
                        node = node.split('|')
                        if idx % 2 == 0:
                            for n_idx, _node in enumerate(node):
                                word = constants.UNK if _node == '' else _node
                                if n_idx == 0:
                                    w, p, s = word.split('\\')
                                    p = 'NN' if p == '' else p
                                    s = str(wn.synsets('entity')[0].offset()) if s == '' else s
                                    _w, position = w.rsplit('_', 1)
                                    # _w = w
                                    words.append(_w)
                                    poses.append(p)
                                    synsets.append(s)
                                    positions.append(min(int(position), constants.MAX_LENGTH))
                                    relations.append(_w)
                                else:
                                    w = word.split('\\')[0]
                        else:
                            rel = '(' + node[0].strip().split('_')[-1]
                            # print(node)
                            # words.append(rel)
                            relations.append(rel)

                    all_words.append(words)
                    all_poses.append(poses)
                    all_synsets.append(synsets)
                    all_positions.append(positions)
                    all_relations.append(relations)
                    all_labels.append([label])
                    all_lens.append([doc_len])
                    all_identities.append((pmid, pair))
            else:
                pass

    # return all_words, all_poses, all_synsets, all_relations, all_labels, all_identities, all_triples, all_positions
    return all_words, all_poses, all_synsets, all_relations, all_labels, all_identities, all_triples, all_lens, \
        all_positions


def parse_sent(raw_data, cid_only=False):
    raw_data = clean_lines(raw_data)
    all_words = []
    all_poses = []
    all_synsets = []
    # all_relations = []
    all_labels = []
    all_identities = []
    all_triples = []
    pmid = ''
    for line in raw_data:
        l = line.strip().split()
        if len(l) == 1:
            pmid = l[0]
        else:
            pair = l[0]
            label = l[1]
            if (label == 'CID' and cid_only) or (not cid_only):
                chem, dis = pair.split('_')
                all_triples.append([chem, dis])

                joint_sdp = ' '.join(l[2:])
                sdps = joint_sdp.split("-PUNC-")
                for sdp in sdps:
                    # S xuoi
                    nodes = sdp.split()
                    words = []
                    poses = []
                    synsets = []
                    # relations = []
                    for idx, node in enumerate(nodes):
                        node = node.split('|')
                        for idx, _node in enumerate(node):
                            word = constants.UNK if _node == '' else _node
                            if idx == 0:
                                w, p, s = word.split('\\')
                                p = 'NN' if p == '' else p
                                s = str(wn.synsets('entity')[0].offset()) if s == '' else s
                                _w, position = w.rsplit('_', 1)
                                # _w = w
                                words.append(_w)
                                poses.append(p)
                                synsets.append(s)
                            else:
                                w = word.split('\\')[0]

                    all_words.append(words)
                    all_poses.append(poses)
                    all_synsets.append(synsets)
                    # all_relations.append(relations)
                    all_labels.append([label])
                    all_identities.append((pmid, pair))
            else:
                pass

    return all_words, all_poses, all_synsets, all_labels, all_identities, all_triples


class Dataset:
    def __init__(self, data_name, sdp_name, vocab_words=None, vocab_poses=None, vocab_synset=None, vocab_rels=None,
                 vocab_chems=None,
                 vocab_dis=None,
                 process_data='cid'):
        self.data_name = data_name
        self.sdp_name = sdp_name

        self.vocab_words = vocab_words
        self.vocab_poses = vocab_poses
        self.vocab_synsets = vocab_synset
        self.vocab_rels = vocab_rels

        self.vocab_chems = vocab_chems
        self.vocab_dis = vocab_dis

        if process_data:
            self._process_data()
            self._clean_data()

    def get_padded_data(self, shuffled=True):
        return self._pad_data(shuffled=shuffled)

    def _clean_data(self):
        del self.vocab_poses
        del self.vocab_synsets
        del self.vocab_rels

    def _process_data(self):
        with open(self.data_name, 'r') as f1:
            raw_data = f1.readlines()
        with open(self.sdp_name, 'r') as f1:
            raw_sdp = f1.readlines()
        data_words, data_pos, data_synsets, data_relations, data_y, self.identities, data_triples, data_lens,\
            data_positions = parse_words(raw_sdp)
        data_words_full, data_pos, data_synsets, data_y, self.identities, data_triples = parse_sent(
            raw_data)

        words = []
        head_mask = []
        e1_mask = []
        e2_mask = []
        labels = []
        poses = []
        synsets = []
        relations = []
        all_ents = []
        # max_len_word = 0

        for tokens in data_words_full:
            sdp_sent = ' '.join(tokens)
            token_ids = constants.tokenizer.encode(sdp_sent)
            words.append(token_ids)

            e1_ids, e2_ids, e1_ide, e2_ide = None, None, None, None
            for i in range(len(token_ids)):
                if token_ids[i] == constants.START_E1:
                    e1_ids = i
                if token_ids[i] == constants.END_E1:
                    e1_ide = i
                if token_ids[i] == constants.START_E2:
                    e2_ids = i
                if token_ids[i] == constants.END_E2:
                    e2_ide = i
            pos = [e1_ids, e1_ide, e2_ids, e2_ide]
            all_ents.append(pos)

        for t in all_ents:
            m0 = []
            for i in range(constants.MAX_LENGTH):
                m0.append(0.0)
            m0[0] = 1.0
            head_mask.append(m0)
            m1 = []
            for i in range(constants.MAX_LENGTH):
                m1.append(0.0)
            for i in range(t[0], t[1] - 1):
                m1[i] = 1 / (t[1] - 1 - t[0])
            e1_mask.append(m1)
            m2 = []
            for i in range(constants.MAX_LENGTH):
                m2.append(0.0)
            for i in range(t[2] - 2, t[3] - 3):
                m2[i] = 1 / ((t[3] - 3) - (t[2] - 2))
            e2_mask.append(m2)

        for i in range(len(data_pos)):

            ps, ss = [], []

            for p, s in zip(data_pos[i], data_synsets[i]):
                if p in self.vocab_poses:
                    p_id = self.vocab_poses[p]
                else:
                    p_id = self.vocab_poses['NN']
                ps += [p_id]
                if s in self.vocab_synsets:
                    synset_id = self.vocab_synsets[s]
                else:
                    synset_id = self.vocab_synsets[constants.UNK]
                ss += [synset_id]

            poses.append(ps)
            synsets.append(ss)

            lb = constants.ALL_LABELS.index(data_y[i][0])
            labels.append(lb)

        for i in range(len(data_relations)):
            rs = []
            for r in data_relations[i]:
                if data_relations[i].index(r) % 2 == 0:
                    if r in self.vocab_words:
                        r_id = self.vocab_words[r]
                    else:
                        r_id = self.vocab_words[constants.UNK]
                else:
                    if r in self.vocab_rels:
                        r_id = len(self.vocab_words) + self.vocab_rels[r] + 1
                    else:
                        r_id = len(self.vocab_words) + self.vocab_rels[constants.UNK] + 1
                rs.append(r_id)
            relations.append(rs)

        self.words = words
        self.head_mask = head_mask
        # print(self.head_mask)
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.relations = relations
        # print(max([len(u) for u in self.relations]))
        self.labels = labels
        self.poses = poses
        self.synsets = synsets
        # self.positions_1 = positions_1
        # self.positions_2 = positions_2
        self.triples = self.parse_triple(data_triples)
        # self.triples = np.zeros([len(self.words), 2])

    def parse_triple(self, all_triples):
        data_triples = []
        for c, d in all_triples:
            if c in self.vocab_chems and d in self.vocab_dis:
                c_id = int(self.vocab_chems[c])
                d_id = int(self.vocab_dis[d]) + int(len(self.vocab_chems))
            elif c in self.vocab_chems:
                c_id = int(self.vocab_chems[c])
                d_id = int(len(self.vocab_dis)) + int(len(self.vocab_chems))
            else:
                c_id = int(len(self.vocab_chems))
                d_id = int(len(self.vocab_dis)) + int(len(self.vocab_chems))
            data_triples.append([c_id, d_id])

        return data_triples

    def _pad_data(self, shuffled=True):
        if shuffled:
            word_shuffled, head_shuffle, e1_shuffle, e2_shuffle, pos_shuffled, synset_shuffled, relation_shuffled, \
            label_shuffled, triple_shuffled = shuffle(
                self.words, self.head_mask, self.e1_mask, self.e2_mask, self.poses,
                self.synsets, self.relations, self.labels, self.triples)
        else:
            word_shuffled, head_shuffle, e1_shuffle, e2_shuffle, pos_shuffled, synset_shuffled, relation_shuffled, \
            label_shuffled, triple_shuffled = \
                self.words, self.head_mask, self.e1_mask, self.e2_mask, self.poses, \
                self.synsets, self.relations, self.labels, self.triples

        self.words = tf.constant(pad_sequences(word_shuffled, maxlen=constants.MAX_LENGTH, padding='post'))
        self.poses = tf.constant(pad_sequences(pos_shuffled, maxlen=constants.MAX_LENGTH, padding='post'))
        self.synsets = tf.constant(pad_sequences(synset_shuffled, maxlen=constants.MAX_LENGTH, padding='post'))
        self.relations = tf.constant(pad_sequences(relation_shuffled, maxlen=50, padding='post'))
        num_classes = len(constants.ALL_LABELS)
        self.labels = tf.keras.utils.to_categorical(label_shuffled, num_classes=num_classes)
        # self.labels = tf.constant(label_shuffled, dtype='float32')
        self.triples = tf.constant(triple_shuffled, dtype='int32')
        self.head_mask = tf.constant(head_shuffle, dtype='float32')
        self.e1_mask = tf.constant(e1_shuffle, dtype='float32')
        self.e2_mask = tf.constant(e2_shuffle, dtype='float32')


class CLDataset:
    def __init__(self, data_name, sdp_name, process_data=True):
        self.data_name = data_name
        self.sdp_name = sdp_name

        if process_data:
            self._process_data()

    def get_padded_data(self, shuffled=True):
        return self._pad_data(shuffled=shuffled)

    def _process_data(self):
        with open(self.data_name, 'r') as f1:
            raw_data = f1.readlines()
        with open(self.sdp_name, 'r') as f1:
            raw_sdp = f1.readlines()
        data_words, data_pos, data_synsets, data_y, self.identities, data_triples = parse_sent(
            raw_data, cid_only=True)
        data_word_sdp, data_pos_sdp, data_synsets_sdp, data_relations, data_y, self.identities, data_triples, data_lens,\
            data_positions_sdp = parse_words(raw_sdp, cid_only=True)

        all_words = []
        all_augments = []

        # print(len(data_words), len(data_synsets), len(data_positions_sdp))

        for words, positions, pos in zip(data_words, data_positions_sdp, data_pos):
            # print(len(words), len(positions), len(synsets))
            augment_number = random.randint(0, (len(words) - len(positions)) // 2)
            sent = ' '.join(words)
            token_ids = constants.tokenizer.encode(sent)
            all_words.append(token_ids)

            augments = []
            start_e1, end_e1, start_e2, end_e2 = 0, 0, 0, 0

            for idx, tok in enumerate(words):
                if '<e1>' in tok:
                    start_e1 = idx
                if '</e1>' in tok:
                    end_e1 = idx
                if '<e2>' in tok:
                    start_e2 = idx
                if '</e2>' in tok:
                    end_e2 = idx

            entity_pos = positions
            for idx, tok in enumerate(words):
                if start_e1 <= idx <= end_e1:
                    entity_pos.append(idx)
                if start_e2 <= idx <= end_e2:
                    entity_pos.append(idx)

            entity_pos = list(set(entity_pos))

            for idx, tok in enumerate(words):
                if idx not in entity_pos and augment_number > 0:
                    if hypernym(tok, pos[idx]):
                        new_word = random.choice(hypernym(tok, pos[idx]))
                    else:
                        new_word = wn.synset('entity.n.01').lemmas()[0].name()
                    new_word = new_word.replace("_", " ").replace("-", " ").lower()
                    new_word = "".join([char for char in new_word if char in ' qwertyuiopasdfghjklzxcvbnm'])
                    augments.append(new_word)
                    print("Replace word: {} with word: {}".format(tok, new_word))
                    augment_number -= 1
                else:
                    augments.append(tok)

            a_sent = ' '.join(augments)
            augment_ids = constants.tokenizer.encode(a_sent)
            all_augments.append(augment_ids)

        self.labels = all_words
        self.augments = all_augments

    def _pad_data(self, shuffled=True):
        if shuffled:
            word_shuffled, label_shuffled = shuffle(self.augments, self.labels)
        else:
            word_shuffled, label_shuffled = self.augments, self.labels

        self.augments = pad_sequences(word_shuffled, maxlen=constants.MAX_LENGTH, padding='post')
        self.labels = pad_sequences(label_shuffled, maxlen=constants.MAX_LENGTH, padding='post')

        sequence_dict = {
            'augments': self.augments,
            'labels': self.labels
        }
        return sequence_dict

