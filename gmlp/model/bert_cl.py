import tensorflow as tf
from tensorflow_addons.losses import ContrastiveLoss
import constants
from constants import *
import os


class BertCLModel:
    def __init__(self, base_encoder):
        if not os.path.exists(TRAINED_MODELS):
            os.makedirs(TRAINED_MODELS)

        self.encoder = base_encoder
        self.max_length = constants.MAX_LENGTH
        self.trained_models = constants.TRAINED_CL

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
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                          patience=constants.PATIENCE)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=TRAINED_MODELS,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        self.model.fit(x=train_data.augments,
                       y=train_data.labels,
                       validation_data=(val_data.augments, val_data.labels),
                       batch_size=16, epochs=constants.EPOCHS, callbacks=[early_stopping, model_checkpoint_callback])

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

