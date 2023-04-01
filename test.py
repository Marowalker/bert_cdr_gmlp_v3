from dataset import parse_words, Dataset
import constants
from data_utils import make_triple_vocab, load_vocab, get_trimmed_w2v_vectors, dot_simililarity_dim1, dot_simililarity_dim2, get_negative_mask
import pickle
import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import TFAutoModel, AutoTokenizer

# vocab_poses = load_vocab(constants.ALL_POSES)
# vocab_synsets = load_vocab(constants.ALL_SYNSETS)
# vocab_rels = load_vocab(constants.ALL_DEPENDS)
# vocab_words = load_vocab(constants.ALL_WORDS)
#
# chem_vocab = make_triple_vocab(constants.DATA + 'chemical2id.txt')
# dis_vocab = make_triple_vocab(constants.DATA + 'disease2id.txt')
#
# with open(constants.RAW_DATA + 'sdp_data_acentors_full.train.txt') as f:
#     lines = f.readlines()

# all_words, all_poses, all_synsets, all_relations, all_labels, all_identities, all_triples, all_lens, all_positions = \
#     parse_words(raw_data=lines)
# for po, le in zip(all_words, all_lens):
#     print(po, le)


# train = Dataset(constants.RAW_DATA + 'sentence_data_acentors_full.train.txt',
#                 constants.RAW_DATA + 'sdp_data_acentors_full.train.txt',
#                 vocab_words=vocab_words,
#                 vocab_poses=vocab_poses,
#                 vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab)
# # pickle.dump(train, open(constants.PICKLE_DATA + 'train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
# #
# dev = Dataset(constants.RAW_DATA + 'sentence_data_acentors_full.dev.txt',
#               constants.RAW_DATA + 'sdp_data_acentors_full.dev.txt',
#               vocab_words=vocab_words,
#               vocab_poses=vocab_poses,
#               vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab, )
# # pickle.dump(dev, open(constants.PICKLE_DATA + 'dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
#
# test = Dataset(constants.RAW_DATA + 'sentence_data_acentors_full.test.txt',
#                constants.RAW_DATA + 'sdp_data_acentors_full.test.txt',
#                vocab_words=vocab_words,
#                vocab_poses=vocab_poses,
#                vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab, )
# pickle.dump(test, open(constants.PICKLE_DATA + 'test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

# Train, Validation Split
# validation = Dataset('', '', process_data=None)
# train_ratio = 0.85
# n_sample = int(len(dev.words) * (2 * train_ratio - 1))
# props = ['words', 'head_mask', 'e1_mask', 'e2_mask', 'relations', 'labels', 'poses', 'synsets', 'identities',
#          'triples']
#
# for prop in props:
#     train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
#     validation.__dict__[prop] = dev.__dict__[prop][n_sample:]
#
# len_train = max([len(w) for w in train.words])
# len_val = max([len(w) for w in validation.words])
# len_test = max([len(w) for w in test.words])
#
# print(max([len_train, len_val, len_test]))

# train.get_padded_data()
# validation.get_padded_data()
#
# print(train.e1_mask)

# wn_emb = get_trimmed_w2v_vectors('data/w2v_model/wordnet_embeddings.npz')
#
# # print(wn_emb.shape)
#
# with open(constants.EMBEDDING_CHEM, 'rb') as f:
#     chem_emb = pickle.load(f)
#     f.close()
#
# with open(constants.EMBEDDING_DIS, 'rb') as f:
#     dis_emb = pickle.load(f)
#     f.close()
#
# concated = tf.concat([chem_emb, dis_emb], axis=0)
# print(concated)

test_1 = tf.random.uniform(shape=(8, 5, 10))
test_2 = tf.random.uniform(shape=(8, 5, 10))

test_1 = tf.nn.l2_normalize(test_1, axis=1)
test_2 = tf.nn.l2_normalize(test_2, axis=1)

dim1_test = dot_simililarity_dim1(test_1, test_2)
dim1_test = tf.reshape(dim1_test, (8, 1))
dim1_test /= 0.1
# print(dim1_test)

neg = tf.concat([test_2, test_1], axis=0)

loss = 0

cr = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,)

for t in [test_1, test_2]:
    dim2_test = dot_simililarity_dim2(t, neg)
    labels = tf.zeros(8, dtype=tf.int32)

    bool_mask = get_negative_mask(8)
    dim2_test = tf.boolean_mask(dim2_test, bool_mask)

    dim2_test = tf.reshape(dim2_test, (8, -1))
    dim2_test /= 0.1

    logits = tf.concat([dim1_test, dim2_test], axis=1)
    loss += cr(y_pred=logits, y_true=labels)

loss /= 16
print(loss)



