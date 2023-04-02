import tensorflow as tf
import tensorflow_addons as tfa
import constants
from constants import *
import os
from utils import Timer, Log
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_utils import *


def contrastive_loss(y_true, y_pred, batch_size=8, t=0.1):
    test_1 = y_pred
    test_2 = y_true

    test_1 = tf.nn.l2_normalize(test_1, axis=1)
    test_2 = tf.nn.l2_normalize(test_2, axis=1)

    dim1_test = dot_simililarity_dim1(test_1, test_2)
    dim1_test = tf.reshape(dim1_test, (batch_size, 1))
    dim1_test /= 0.1
    neg = tf.concat([test_2, test_1], axis=0)

    loss = 0

    cr = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, )

    for t in [test_1, test_2]:
        dim2_test = dot_simililarity_dim2(t, neg)
        labels = tf.zeros(batch_size, dtype=tf.int32)

        bool_mask = get_negative_mask(batch_size)
        dim2_test = tf.boolean_mask(dim2_test, bool_mask)

        dim2_test = tf.reshape(dim2_test, (batch_size, -1))
        dim2_test /= 0.1

        logits = tf.concat([dim1_test, dim2_test], axis=1)
        loss += cr(y_pred=logits, y_true=labels)

    loss /= (2 * batch_size)
    return loss


class BertCLModel:
    def __init__(self, base_encoder):
        self.encoder = base_encoder
        self.max_length = constants.MAX_LENGTH
        self.trained_models = constants.TRAINED_CL

        if not os.path.exists(self.trained_models):
            os.makedirs(self.trained_models)

    def _add_inputs(self):
        self.input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype='int32')

    def _bert_layer(self):
        self.bertoutput = self.encoder(self.input_ids)
        emb = self.bertoutput[0]
        out = tf.keras.layers.Dense(constants.INPUT_W2V_DIM, activation='relu')(emb)
        return out

    def _add_train_ops(self):
        self.model = tf.keras.Model(
            inputs=self.input_ids,
            outputs=self._bert_layer())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # self.model.compile(optimizer=self.optimizer, loss=custom_contrastive_loss)
        # print(self.model.summary())

    def _train(self, train_data, val_data):
        best_loss = 100000
        n_epoch_no_improvement = 0

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(8)

        val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
        val_dataset = val_dataset.batch(8)

        for e in range(constants.EPOCHS):
            print("\nStart of epoch %d" % (e + 1,))

            # Iterate over the batches of the dataset.
            for idx, batch in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(batch['augments'], training=True)
                    labels = self.model(batch['labels'], training=True)

                    loss_value = contrastive_loss(labels, logits, batch_size=len(batch['augments']))
                    # loss_value = tf.reduce_mean(loss_value)

                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                    self.optimizer.apply_gradients((grad, var) for (grad, var) in
                                                   zip(grads, self.model.trainable_variables) if grad is not None)
                    if idx % 500 == 0:
                        Log.log("Iter {}, Loss: {} ".format(idx, loss_value))

            if constants.EARLY_STOPPING:
                total_loss = []

                for idx, batch in enumerate(val_dataset):
                    val_logits = self.model(batch['augments'], training=True)
                    val_labels = self.model(batch['labels'], training=True)

                    v_loss = contrastive_loss(val_labels, val_logits, batch_size=len(batch['augments']))
                    # v_loss = tf.reduce_mean(v_loss)
                    total_loss.append(float(v_loss))

                val_loss = np.mean(total_loss)
                Log.log("Loss at epoch number {}: {}".format(e + 1, val_loss))
                print("Previous best loss: ", best_loss)

                if val_loss < best_loss:
                    Log.log('Save the model at epoch {}'.format(e + 1))
                    self.model.save_weights(self.trained_models)
                    best_loss = val_loss
                    n_epoch_no_improvement = 0

                else:
                    n_epoch_no_improvement += 1
                    Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
                    if n_epoch_no_improvement >= constants.PATIENCE:
                        print("Best loss: {}".format(best_loss))
                        break

            if not constants.EARLY_STOPPING:
                self.model.save_weights(self.trained_models)

        # self.model.save_weights(TRAINED_MODELS)

    def plot_model(self):
        tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=False, show_layer_names=True,
                                  rankdir='TB',
                                  expand_nested=False, dpi=300)

    def build(self, train_data=None, val_data=None, training=True):
        with tf.device('/device:GPU:0'):
            self._add_inputs()
            self._add_train_ops()
            if training:
                self._train(train_data, val_data)

    def get_embeddings(self, test_data, training=None):
        self.model.load_weights(self.trained_models)
        # pred = self.model.predict(test_data)
        pred = self.encoder(test_data)[0]
        return pred
