import tensorflow as tf
import tensorflow_addons as tfa
import constants
from constants import *
import os
from utils import Timer, Log
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def contrastive_loss(y_true, y_pred, t=0.1):
    loss_value = 0
    for i in range(len(y_true)):
        z_t = y_true[i]
        z_p = y_pred[i]
        cos_pos = tf.math.exp(tf.keras.losses.CosineSimilarity()(z_t, z_p).numpy() / t)
        cos_pos_rev = tf.math.exp(tf.keras.losses.CosineSimilarity()(z_p, z_t).numpy() / t)
        all_neg = []
        all_neg_rev = []
        for idx, (z_i, z_k) in enumerate(zip(y_true, y_pred)):
            cos_neg = tf.math.exp((tf.keras.losses.CosineSimilarity()(z_i, z_k).numpy() / t))
            cos_neg_rev = tf.math.exp((tf.keras.losses.CosineSimilarity()(z_k, z_i).numpy() / t))
            if idx == i:
                all_neg.append(0)
                all_neg_rev.append(0)
            else:
                all_neg.append(cos_neg)
                all_neg_rev.append(cos_neg_rev)
        batch_loss = tf.math.log(cos_pos / sum(all_neg))
        batch_loss_rev = tf.math.log(cos_pos_rev / sum(all_neg_rev))
        loss_value = loss_value - batch_loss - batch_loss_rev
    loss_value /= (2 * float(len(y_true)))

    return loss_value


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

        for e in range(constants.EPOCHS):
            print("\nStart of epoch %d" % (e + 1,))

            train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(16)

            # Iterate over the batches of the dataset.
            for idx, batch in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(batch['augments'], training=True)
                    labels = self.model(batch['labels'], training=True)

                    loss_value = contrastive_loss(labels.numpy(), logits.numpy())
                    # loss_value = tf.reduce_mean(loss_value)

                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                    self.optimizer.apply_gradients((grad, var) for (grad, var) in
                                                   zip(grads, self.model.trainable_variables) if grad is not None)
                    if idx % 50 == 0:
                        Log.log("Iter {}, Loss: {} ".format(idx, loss_value))

            if constants.EARLY_STOPPING:
                total_loss = []
                val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
                val_dataset = val_dataset.batch(16)

                for idx, batch in enumerate(val_dataset):
                    val_logits = self.model(batch['augments'], training=True)
                    val_labels = self.model(batch['labels'], training=True)

                    v_loss = contrastive_loss(val_labels.numpy(), val_logits.numpy())
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
