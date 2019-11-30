"""This file defines the network architecture for the discriminator."""
import tensorflow as tf
from tensorflow.nn import relu, leaky_relu, tanh, sigmoid
from ..ops import dense, conv3d, get_normalization, nn_conv3d

NORMALIZATION = None # 'batch_norm', 'layer_norm', None
ACTIVATION = leaky_relu # relu, leaky_relu, tanh, sigmoid

class Discriminator:
    def __init__(self, n_tracks, beat_resolution=None, name='Discriminator'):
        self.n_tracks = n_tracks
        self.beat_resolution = beat_resolution
        self.name = name

    def __call__(self, tensor_in, condition=None, training=None):
        norm = get_normalization(NORMALIZATION, training)
        conv_layer = lambda i, f, k, s: ACTIVATION(norm(conv3d(i, f, k, s)))

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h = tensor_in

            # Compute chroma feature
            n_beats = h.get_shape()[2] // self.beat_resolution
            reshaped = tf.reshape(
                tensor_in, (-1, h.get_shape()[1], n_beats, self.beat_resolution,
                            h.get_shape()[3], h.get_shape()[4]))
            summed = tf.reduce_sum(reshaped, 3)
            factor = int(h.get_shape()[3]) // 12
            remainder = int(h.get_shape()[3]) % 12
            reshaped = tf.reshape(
                summed[..., :(factor * 12), :],
                (-1, h.get_shape()[1], n_beats, factor, 12, h.get_shape()[4]))
            chroma = tf.reduce_sum(reshaped, 3)                      # 4, 4, 12
            if remainder:
                chroma += summed[..., -remainder:, :]

            # Compute onset/offset feature
            padded = tf.pad(tensor_in[:, :, :-1],
                            ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
            on_off_set = tf.reduce_sum(tensor_in - padded, 3, True)  # 4, 48, 1

            # Pitch-time private network
            with tf.variable_scope('pitch_time_private'):
                s1 = [conv_layer(h, 16, (1, 1, 12), (1, 1, 12))      # 4, 48, 7
                      for _ in range(self.n_tracks)]
                s1 = [conv_layer(s1[i], 32, (1, 3, 1), (1, 3, 1))    # 4, 16, 7
                      for i in range(self.n_tracks)]

            # Time-pitch private network
            with tf.variable_scope('time_pitch_private'):
                s2 = [conv_layer(h, 16, (1, 3, 1), (1, 3, 1))        # 4, 16, 84
                      for _ in range(self.n_tracks)]
                s2 = [conv_layer(s2[i], 32, (1, 1, 12), (1, 1, 12))  # 4, 16, 7
                      for i in range(self.n_tracks)]

            h = [tf.concat((s1[i], s2[i]), -1) for i in range(self.n_tracks)]

            # Merged private network
            with tf.variable_scope('merged_private'):
                h = [conv_layer(h[i], 64, (1, 1, 1), (1, 1, 1))      # 4, 16, 7
                     for i in range(self.n_tracks)]

            h = tf.concat(h, -1)

            # Shared network
            with tf.variable_scope('shared'):
                h = conv_layer(h, 128, (1, 4, 3), (1, 4, 2))         # 4, 4, 3
                h = conv_layer(h, 256, (1, 4, 3), (1, 4, 3))         # 4, 1, 1

            # Chroma stream
            with tf.variable_scope('chroma'):
                c = conv_layer(chroma, 32, (1, 1, 12), (1, 1, 12))   # 4, 4, 1
                c = conv_layer(c, 64, (1, 4, 1), (1, 4, 1))          # 4, 1, 1

            # Onset/offset stream
            with tf.variable_scope('on_off_set'):
                o = conv_layer(on_off_set, 16, (1, 3, 1), (1, 3, 1)) # 4, 16, 1
                o = conv_layer(o, 32, (1, 4, 1), (1, 4, 1))          # 4, 4, 1
                o = conv_layer(o, 64, (1, 4, 1), (1, 4, 1))          # 4, 1, 1

            h = tf.concat((h, c, o), -1)

            # Merge all streams
            with tf.variable_scope('merged'):
                h = conv_layer(h, 512, (2, 1, 1), (1, 1, 1))         # 3, 1, 1

            h = tf.reshape(h, (-1, h.get_shape()[-1]))
            h = dense(h, 1)

        return h


    def forward_with_given_weights(self, weights, tensor_in, condition=None, training=None):
        norm = get_normalization(NORMALIZATION, training)
        conv_layer = lambda i, f, k, s: ACTIVATION(norm(nn_conv3d(i, f, k, (1,) + s + (1,) )))

        

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h = tensor_in

            # Compute chroma feature
            n_beats = h.get_shape()[2] // self.beat_resolution
            # print("discriminator h", h, "h.get_shape()", h.get_shape())
            reshaped = tf.reshape(
                tensor_in, (-1, h.get_shape()[1], n_beats, self.beat_resolution,
                            h.get_shape()[3], h.get_shape()[4]))
            summed = tf.reduce_sum(reshaped, 3)
            factor = int(h.get_shape()[3]) // 12
            remainder = int(h.get_shape()[3]) % 12
            reshaped = tf.reshape(
                summed[..., :(factor * 12), :],
                (-1, h.get_shape()[1], n_beats, factor, 12, h.get_shape()[4]))
            chroma = tf.reduce_sum(reshaped, 3)                      # 4, 4, 12
            if remainder:
                chroma += summed[..., -remainder:, :]

            # Compute onset/offset feature
            padded = tf.pad(tensor_in[:, :, :-1],
                            ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
            on_off_set = tf.reduce_sum(tensor_in - padded, 3, True)  # 4, 48, 1

            # Pitch-time private network
            with tf.variable_scope('pitch_time_private') as cur_scope:
                # print("cur_scope_name:", cur_scope.name, "cur_scope", cur_scope, type(cur_scope))
                # cur_scopev name: Model/Discriminator/pitch_time_private
                s1_temp = []
                # s1 = [conv_layer(h, 16, (1, 1, 12), (1, 1, 12))      # 4, 48, 7
                #       for _ in range(self.n_tracks)]
                for i in range(self.n_tracks):
                    # need nn_conv3d layers
                    subname = '/conv3d/' if i ==0 else '/conv3d_{}/'.format(i)
                    kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                    # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                    bias_var = weights[cur_scope.name + subname + 'bias:0']
                    # del weights[cur_scope.name + subname + 'bias:0']
                    # print("loading:", kernal_var, bias_var) 
                    s1_temp.append(bias_var + conv_layer(h, kernal_var, (1, 1, 12), (1, 1, 12)) )      # 4, 48, 7
                    

                
                s1 = []
                # s1 = [conv_layer(s1[i], 32, (1, 3, 1), (1, 3, 1))    # 4, 16, 7
                #       for i in range(self.n_tracks)]
                for i in range(self.n_tracks, 2*self.n_tracks):
                    subname = '/conv3d/' if i ==0 else '/conv3d_{}/'.format(i)
                    kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                    # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                    bias_var = weights[cur_scope.name + subname + 'bias:0']
                    # del weights[cur_scope.name + subname + 'bias:0']
                    # print("loading:", kernal_var, bias_var)
                    s1.append(bias_var + conv_layer(s1_temp[i - self.n_tracks], kernal_var, (1, 3, 1), (1, 3, 1)) )    # 4, 16, 7
                del s1_temp 

                

            # Time-pitch private network
            with tf.variable_scope('time_pitch_private') as cur_scope:
                # print("cur_scope_name:", cur_scope.name, "cur_scope", cur_scope, type(cur_scope))
                # s2 = [conv_layer(h, 16, (1, 3, 1), (1, 3, 1))        # 4, 16, 84
                #       for _ in range(self.n_tracks)]
                s2_temp = []
                for i in range(self.n_tracks):
                    # need nn_conv3d layers
                    subname = '/conv3d/' if i ==0 else '/conv3d_{}/'.format(i)
                    kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                    # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                    bias_var = weights[cur_scope.name + subname + 'bias:0']
                    # del weights[cur_scope.name + subname + 'bias:0']
                    # print("loading:", kernal_var, bias_var)
                    s2_temp.append(bias_var + conv_layer(h, kernal_var, (1, 3, 1), (1, 3, 1)) )       # 4, 16, 84

                # s2 = [conv_layer(s2[i], 32, (1, 1, 12), (1, 1, 12))  # 4, 16, 7
                #       for i in range(self.n_tracks)]
                s2 = []
                for i in range(self.n_tracks, 2*self.n_tracks):
                    subname = '/conv3d/' if i ==0 else '/conv3d_{}/'.format(i)
                    kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                    # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                    bias_var = weights[cur_scope.name + subname + 'bias:0']
                    # del weights[cur_scope.name + subname + 'bias:0']
                    # print("loading:", kernal_var, bias_var)
                    s2.append(bias_var + conv_layer(s2_temp[i - self.n_tracks], kernal_var, (1, 1, 12), (1, 1, 12)) ) # 4, 16, 7
                del s2_temp

            h = [tf.concat((s1[i], s2[i]), -1) for i in range(self.n_tracks)]

            # Merged private network
            with tf.variable_scope('merged_private') as cur_scope:
                # h = [conv_layer(h[i], 64, (1, 1, 1), (1, 1, 1))      # 4, 16, 7
                #      for i in range(self.n_tracks)]
                old_h = h 
                h = [] 
                for i in range(self.n_tracks): 
                    subname = '/conv3d/' if i ==0 else '/conv3d_{}/'.format(i)
                    kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                    # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                    bias_var = weights[cur_scope.name + subname + 'bias:0']
                    # del weights[cur_scope.name + subname + 'bias:0']
                    # print("loading:", kernal_var, bias_var)
                    h.append(bias_var + conv_layer(old_h[i], kernal_var, (1, 1, 1), (1, 1, 1)) )      # 4, 16, 7
                del old_h

            h = tf.concat(h, -1)


            # Shared network
            with tf.variable_scope('shared') as cur_scope:
                subname = '/conv3d/'
                kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                bias_var = weights[cur_scope.name + subname + 'bias:0']
                # del weights[cur_scope.name + subname + 'bias:0']
                # print("loading:", kernal_var, bias_var)
                # h = conv_layer(h, 128, (1, 4, 3), (1, 4, 2))         # 4, 4, 3
                h = bias_var + conv_layer(h, kernal_var, (1, 4, 3), (1, 4, 2))         # 4, 4, 3

                subname = '/conv3d_1/'
                kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                bias_var = weights[cur_scope.name + subname + 'bias:0']
                # del weights[cur_scope.name + subname + 'bias:0']
                # print("loading:", kernal_var, bias_var)
                # h = conv_layer(h, 256, (1, 4, 3), (1, 4, 3))         # 4, 1, 1
                h = bias_var + conv_layer(h, kernal_var, (1, 4, 3), (1, 4, 3))         # 4, 1, 1


            # Chroma stream
            with tf.variable_scope('chroma') as cur_scope:
                subname = '/conv3d/'
                kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                bias_var = weights[cur_scope.name + subname + 'bias:0']
                # del weights[cur_scope.name + subname + 'bias:0']
                # print("loading:", kernal_var, bias_var)
                # c = conv_layer(chroma, 32, (1, 1, 12), (1, 1, 12))   # 4, 4, 1
                c = bias_var + conv_layer(chroma, kernal_var, (1, 1, 12), (1, 1, 12))   # 4, 4, 1
                
                subname = '/conv3d_1/'
                kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                bias_var = weights[cur_scope.name + subname + 'bias:0']
                # del weights[cur_scope.name + subname + 'bias:0']
                # print("loading:", kernal_var, bias_var)         
                # c = conv_layer(c, 64, (1, 4, 1), (1, 4, 1))          # 4, 1, 1
                c = bias_var + conv_layer(c, kernal_var, (1, 4, 1), (1, 4, 1))          # 4, 1, 1


            # Onset/offset stream
            with tf.variable_scope('on_off_set') as cur_scope:
                subname = '/conv3d/'
                kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                bias_var = weights[cur_scope.name + subname + 'bias:0']
                # del weights[cur_scope.name + subname + 'bias:0']
                # print("loading:", kernal_var, bias_var)
                # o = conv_layer(on_off_set, 16, (1, 3, 1), (1, 3, 1)) # 4, 16, 1
                o = bias_var + conv_layer(on_off_set, kernal_var, (1, 3, 1), (1, 3, 1)) # 4, 16, 1

                subname = '/conv3d_1/'
                kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                bias_var = weights[cur_scope.name + subname + 'bias:0']
                # del weights[cur_scope.name + subname + 'bias:0']
                # print("loading:", kernal_var, bias_var)
                # o = conv_layer(o, 32, (1, 4, 1), (1, 4, 1))          # 4, 4, 1
                o = bias_var + conv_layer(o, kernal_var, (1, 4, 1), (1, 4, 1))          # 4, 4, 1

                subname = '/conv3d_2/'
                kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                bias_var = weights[cur_scope.name + subname + 'bias:0']
                # del weights[cur_scope.name + subname + 'bias:0']
                # print("loading:", kernal_var, bias_var)
                # o = conv_layer(o, 64, (1, 4, 1), (1, 4, 1))          # 4, 1, 1
                o = bias_var + conv_layer(o, kernal_var, (1, 4, 1), (1, 4, 1))          # 4, 1, 1


            h = tf.concat((h, c, o), -1)

            # Merge all streams
            with tf.variable_scope('merged') as cur_scope:
                subname = '/conv3d/'
                kernal_var = weights[cur_scope.name + subname + 'kernel:0']
                # del weights[cur_scope.name + subname + 'kernel:0'] # del only for debugging purpose 
                bias_var = weights[cur_scope.name + subname + 'bias:0']
                # del weights[cur_scope.name + subname + 'bias:0']
                # print("loading:", kernal_var, bias_var)
                # h = conv_layer(h, 512, (2, 1, 1), (1, 1, 1))         # 3, 1, 1
                h = bias_var + conv_layer(h, kernal_var, (2, 1, 1), (1, 1, 1))         # 3, 1, 1


            h = tf.reshape(h, (-1, h.get_shape()[-1]))
            # h = dense(h, 1)
            kernal_var = weights['Model/Discriminator/dense/kernel:0']
            # del weights['Model/Discriminator/dense/kernel:0'] # del only for debugging purpose 
            bias_var = weights['Model/Discriminator/dense/bias:0']
            # del weights['Model/Discriminator/dense/bias:0']
            
            # print("last h",h, "kernal_var",kernal_var, "bias_var",bias_var)
            h = bias_var + tf.matmul(h, kernal_var)
            # print("get h:", h)
                

        return h
