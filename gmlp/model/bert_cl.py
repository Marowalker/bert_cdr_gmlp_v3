import tensorflow as tf
from tensorflow_addons.losses import ContrastiveLoss
import constants
from constants import *
import os
from utils import Timer, Log
import numpy as np


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
        out = tf.keras.layers.Dense(constants.INPUT_W2V_DIM, activation='softmax')(emb)
        return out

    def _add_train_ops(self):
        self.model = tf.keras.Model(
            inputs=self.input_ids,
            outputs=self._bert_layer())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=4e-6)
        self.model.compile(optimizer=self.optimizer,
                           loss=ContrastiveLoss())
        print(self.model.summary())

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
                    labels = self.model(batch['labels'])

                    loss_value = ContrastiveLoss(labels, logits)

                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    if idx % 500 == 0:
                        Log.log("Iter {}, Loss: {} ".format(idx, loss_value))

                if constants.EARLY_STOPPING:
                    total_loss = []
                    val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
                    val_dataset = val_dataset.batch(16)

                    for idx, batch in enumerate(val_dataset):
                        val_logits = self.model(batch['augments'], training=False)
                        val_labels = self.model(batch['labels'])

                        v_loss = ContrastiveLoss(val_labels, val_logits)
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

    def build(self, train_data, val_data, training=True):
        with tf.device('/device:GPU:0'):
            self._add_inputs()
            self._add_train_ops()
            if training:
                self._train(train_data, val_data)

    def get_emeddings(self, test_data, training=None):
        self.model.load_weights(self.trained_models)
        pred = self.model(test_data, training=training)
        return pred

