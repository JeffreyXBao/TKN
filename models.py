import tensorflow as tf
import numpy as np
import tkn

class TKN_Basic(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_dim, output_dim, maximum_position_encoding, rate=0.1):
        super(TKN_Basic, self).__init__()
        self.input_dim = input_dim

        #self.tkn_layer = tkn.TKN(num_layers, d_model, num_heads, dff, input_dim, output_dim, maximum_position_encoding)

        #TODO set d_model to input_dim (because embedding not working yet), should be set to embedding output dim
        self.tkn_layer = tkn.TKN(num_layers, input_dim, num_heads, dff, input_dim, output_dim, maximum_position_encoding)

        self.final_layer = tf.keras.layers.Dense(output_dim)
    def call(self, inp, training):
        tkn_output = self.tkn_layer(inp, training)  # (batch_size, inp_seq_len, d_model)
        final_output = self.final_layer(tkn_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output
