from __future__ import division, print_function
import codecs
import pickle
import numpy as np
from keras.models import Model, Input
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D, Activation, Dropout, \
    Flatten, Dense, add, concatenate, multiply, Permute, Reshape, RepeatVector, Embedding, TimeDistributed, LSTM, \
    Bidirectional, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import plot_model
from data import load_coco_data, sample_coco_minibatch, decode_captions


class ImageCaptionModel(object):
    def __init__(self, feature_dim=512, embed_dim=300, z_dim=300, lstm_dim=512, sent_len=17, vocab_size=1004,
                 dp_rate=0.9, batch_size=100):
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.z_dim = z_dim
        self.lstm_dim = lstm_dim
        self.sent_len = sent_len
        self.vocab_size = vocab_size
        self.dp_rate = dp_rate
        self.batch_size = batch_size
    
    def _image_model(self):
        image_input = Input((self.feature_dim,), dtype='float32')
        im_att_1 = Dense(self.feature_dim, activation='softmax')(image_input)
        weighted_im_input = multiply([image_input, im_att_1])
        weighted_im_input = Reshape((self.feature_dim, 1))(weighted_im_input)
        
        conv_1 = Conv1D(32, 3, strides=1, padding='same', dilation_rate=1, activation='relu', name='conv_1')(
            weighted_im_input)
        conv_1 = MaxPooling1D(2, 2)(conv_1)
        conv_1 = Dropout(self.dp_rate)(conv_1)
        temp_conv_1 = Flatten()(Conv1D(1, 1)(conv_1))
        im_att_2_1 = Dense(int(self.feature_dim / 2), activation='softmax')(image_input)
        im_att_2_2 = Dense(int(self.feature_dim / 2), activation='softmax')(temp_conv_1)
        im_att_2 = add([im_att_2_1, im_att_2_2])
        im_att_2 = Reshape((-1, 1))(im_att_2)
        weighted_conv_1 = multiply([im_att_2, conv_1])
        
        conv_2 = Conv1D(128, 3, strides=1, padding='same', dilation_rate=1, activation='relu', name='conv_2')(
            weighted_conv_1)
        conv_2 = MaxPooling1D(2, 2)(conv_2)
        conv_2 = Dropout(self.dp_rate)(conv_2)
        temp_conv_2 = Flatten()(Conv1D(1, 1)(conv_2))
        im_att_3_1 = Dense(int(self.feature_dim / 4), activation='softmax')(image_input)
        im_att_3_2 = Dense(int(self.feature_dim / 4), activation='softmax')(temp_conv_1)
        im_att_3_3 = Dense(int(self.feature_dim / 4), activation='softmax')(temp_conv_2)
        im_att_3 = add([im_att_3_1, im_att_3_2, im_att_3_3])
        im_att_3 = Reshape((-1, 1))(im_att_3)
        weighted_conv_2 = multiply([im_att_3, conv_2])
        
        conv_3 = Conv1D(512, 3, strides=1, padding='same', dilation_rate=1, activation='relu', name='conv_3')(
            weighted_conv_2)
        conv_3 = MaxPooling1D(2, 2)(conv_3)
        conv_3 = Dropout(self.dp_rate)(conv_3)
        temp_conv_3 = Flatten()(Conv1D(1, 1)(conv_3))
        im_att_4_1 = Dense(int(self.feature_dim / 8), activation='softmax')(image_input)
        im_att_4_2 = Dense(int(self.feature_dim / 8), activation='softmax')(temp_conv_1)
        im_att_4_3 = Dense(int(self.feature_dim / 8), activation='softmax')(temp_conv_2)
        im_att_4_4 = Dense(int(self.feature_dim / 8), activation='softmax')(temp_conv_3)
        im_att_4 = add([im_att_4_1, im_att_4_2, im_att_4_3, im_att_4_4])
        im_att_4 = Reshape((-1, 1))(im_att_4)
        weighted_conv_3 = multiply([im_att_4, conv_3])
        
        model = Model(inputs=[image_input], outputs=[weighted_conv_3], name='image_model')
        
        return model
    
    def _caption_model(self, image_feats, prev_words):
        Vg = GlobalAveragePooling1D()(image_feats)
        Vg = Dense(self.embed_dim, activation='relu')(Vg)
        Vg = Dropout(self.dp_rate)(Vg)
        
        Vi = Conv1D(self.z_dim, 1, padding='same', activation='relu')(image_feats)
        Vi = Dropout(self.dp_rate)(Vi)
        Vi_embed = Conv1D(self.embed_dim, 1, padding='same', activation='relu')(Vi)
        
        x = RepeatVector(self.sent_len)(Vg)
        
        embedding = Embedding(self.vocab_size, self.embed_dim, input_length=self.sent_len)(prev_words)
        embedding = Activation('relu')(embedding)
        embedding = Dropout(self.dp_rate)(embedding)
        
        x = concatenate([x, embedding])
        x = Dropout(self.dp_rate)(x)
        
        h = LSTM(self.lstm_dim, return_sequences=True, dropout=self.dp_rate, name='h_forward')(x)
        num_vfeats = int(self.feature_dim / 8)
        h_out_linear = Conv1D(self.z_dim, 1, padding='same', activation='tanh', name='zh_linear')(h)
        h_out_linear = Dropout(self.dp_rate)(h_out_linear)
        h_out_embed = Conv1D(self.embed_dim, 1, padding='same', name='zh_embed')(h_out_linear)
        z_h_embed = TimeDistributed(RepeatVector(num_vfeats))(h_out_embed)
        z_v_linear = TimeDistributed(RepeatVector(self.sent_len), name='z_v_linear')(Vi)
        z_v_embed = TimeDistributed(RepeatVector(self.sent_len), name='z_v_embed')(Vi_embed)
        z_v_linear = Permute((2, 1, 3))(z_v_linear)
        z_v_embed = Permute((2, 1, 3))(z_v_embed)
        z = add([z_h_embed, z_v_embed])
        z = Dropout(self.dp_rate)(z)
        z = TimeDistributed(Activation('tanh', name='merge_v_h_tanh'))(z)
        
        att = TimeDistributed(Conv1D(1, 1, padding='same'), name='att')(z)
        att = Reshape((self.sent_len, num_vfeats), name='att_res')(att)
        att = TimeDistributed(Activation('softmax'), name='att_scores')(att)
        att = TimeDistributed(RepeatVector(self.z_dim), name='att_rep')(att)
        att = Permute((1, 3, 2), name='att_rep_p')(att)
        
        weighted_vi = multiply([att, z_v_linear])
        c_vec = TimeDistributed(Lambda(lambda a: K.sum(a, axis=-2), output_shape=(self.z_dim,)), name='c_vec')(
            weighted_vi)
        atten_out = add([h_out_linear, c_vec])
        
        h = TimeDistributed(Dense(self.embed_dim, activation='tanh'))(atten_out)
        h = Dropout(self.dp_rate)(h)
        
        predictions = TimeDistributed(Dense(self.vocab_size, activation='softmax'), name='out')(h)
        
        return predictions
    
    def build(self):
        image_input = Input(shape=(self.feature_dim,), name='image_input')
        image_model = self._image_model()
        
        image_feats = image_model(image_input)
        
        conv_feats = Input(shape=(int(self.feature_dim / 8), 512), name='conv_feats')
        prev_words = Input(shape=(self.sent_len,), name='prev_words')
        
        predictions = self._caption_model(image_feats, prev_words)
        
        model = Model(inputs=[image_input, prev_words], outputs=predictions, name='out_model')
        
        return model


if __name__ == '__main__':
    batch_size = 10
    learning_rate = 3e-3
    icm = ImageCaptionModel(feature_dim=512, embed_dim=300, z_dim=300, lstm_dim=512, sent_len=17, vocab_size=1004,
                            dp_rate=0.9, batch_size=batch_size)
    model = icm.build()
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode="temporal",
                  metrics=['mse'])
    model.summary()
    # plot_model(model, to_file='../doc/model.png')
    
    model_file = '../models/weights.{epoch:02d}-{val_loss:.2f}.h5'
    
    ep = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    mc = ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True,
                         mode='auto')
    
    data = load_coco_data('../data/coco_captioning', max_train=None, pca_features=True)
    num_train = batch_size#data['train_captions'].shape[0]
    num_val = batch_size#data['val_captions'].shape[0]
    caps_train, imfeats_train, urls_train = sample_coco_minibatch(data, batch_size=num_train, split='train')
    caps_val, imfeats_val, urls_val = sample_coco_minibatch(data, batch_size=num_val, split='val')
    
    model.fit(x={'image_input': imfeats_train, 'prev_words': caps_train},
              y={'out': caps_train.reshape(num_train, 17, 1)},
              batch_size=batch_size, epochs=10, verbose=1, callbacks=[ep, mc],
              validation_data=(
                  {'image_input': imfeats_val, 'prev_words': caps_val}, {'out': caps_val.reshape(num_val, 17, 1)}))
    
    predictions_train = model.predict(x={'image_input': imfeats_train, 'prev_words': caps_train}, batch_size=batch_size)
    predictions_val = model.predict(x={'image_input': imfeats_val, 'prev_words': caps_val}, batch_size=batch_size)
    mse_train = model.evaluate(x={'image_input': imfeats_train, 'prev_words': caps_train},
                               y={'out': caps_train.reshape(num_train, 17, 1)},
                               batch_size=batch_size)
    mse_val = model.evaluate(x={'image_input': imfeats_val, 'prev_words': caps_val},
                             y={'out': caps_val.reshape(num_val, 17, 1)},
                             batch_size=batch_size)
    model.save('../models/final_model.h5')
    print(predictions_train.argmax(-1)[0])
    print(predictions_val.argmax(-1)[0])
    print(mse_train)
    print(mse_val)
    idx_to_word = data['idx_to_word']
    pickle.dump({'pred_train': predictions_train, 'decode_pred_train': decode_captions(predictions_train, idx_to_word),
                 'pred_val': predictions_val, 'decode_pred_val': decode_captions(predictions_val, idx_to_word)},
                codecs.open('../models/predictions.pkl', 'wb'))
