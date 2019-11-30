"""This file defines the network architecture for the generator."""
import tensorflow as tf
from tensorflow.nn import relu, leaky_relu, tanh, sigmoid
from ..ops import tconv3d, get_normalization, nn_tconv3d

NORMALIZATION = 'batch_norm' # 'batch_norm', 'layer_norm'
ACTIVATION = relu # relu, leaky_relu, tanh, sigmoid

class Generator:
    def __init__(self, n_tracks, name='Generator'):
        self.n_tracks = n_tracks
        self.name = name

        self.batch_size = 64
        self.latent_dim = 128

    def __call__(self, tensor_in, condition=None, training=None, slope=None):
        norm = get_normalization(NORMALIZATION, training)
        tconv_layer = lambda i, f, k, s: ACTIVATION(norm(tconv3d(i, f, k, s)))

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h = tensor_in
            h = tf.expand_dims(tf.expand_dims(tf.expand_dims(h, 1), 1), 1)

            # Shared network
            with tf.variable_scope('shared'):
                h = tconv_layer(h, 512, (4, 1, 1), (4, 1, 1))        # 4, 1, 1
                h = tconv_layer(h, 256, (1, 4, 3), (1, 4, 3))        # 4, 4, 3
                h = tconv_layer(h, 128, (1, 4, 3), (1, 4, 2))        # 4, 16, 7

            # Pitch-time private network
            with tf.variable_scope('pitch_time_private'):
                s1 = [tconv_layer(h, 32, (1, 1, 12), (1, 1, 12))     # 4, 16, 84
                      for _ in range(self.n_tracks)]
                s1 = [tconv_layer(s1[i], 16, (1, 3, 1), (1, 3, 1))   # 4, 48, 84
                      for i in range(self.n_tracks)]

            # Time-pitch private network
            with tf.variable_scope('time_pitch_private'):
                s2 = [tconv_layer(h, 32, (1, 3, 1), (1, 3, 1))       # 4, 48, 7
                      for _ in range(self.n_tracks)]
                s2 = [tconv_layer(s2[i], 16, (1, 1, 12), (1, 1, 12)) # 4, 48, 84
                      for i in range(self.n_tracks)]

            h = [tf.concat((s1[i], s2[i]), -1) for i in range(self.n_tracks)]

            # Merged private network
            with tf.variable_scope('merged_private'):
                h = [norm(tconv3d(h[i], 1, (1, 1, 1), (1, 1, 1)))    # 4, 48, 84
                     for i in range(self.n_tracks)]
                h = tf.concat(h, -1)

        return tanh(h)

    def forward_with_given_weights(self, weights, tensor_in, condition=None, training=None, slope=None):
        norm = get_normalization(NORMALIZATION, training) 
        tconv_layer = lambda i, f, k, s: ACTIVATION(norm(tconv3d(i, f, k, s)))


        def nn_tconv_layer(i, tup, o, s, activation_flag=True):
            # nn_tconv_layer = lambda i, f, o, s: ACTIVATION(norm(nn_tconv3d(i, f, o, (1,) + s + (1,) )))
            f, bias, gamma, beta = tup # unpack
            # print("i", i,"f", f)
            __tmp = nn_tconv3d(i, f, o, (1,) + s + (1,))
            # print("__tmp",__tmp,"tf.shape(__tmp)[:-1]",tf.shape(__tmp)[:-1], tf.shape(__tmp) )
            __mean, __variance = tf.nn.moments(__tmp, [-1],#tf.shape(__tmp)[:-1], 
            name='moments', keep_dims=True) 
            # print("__tmp",__tmp,"__mean",__mean, "__variance", __variance)
            __tmp = tf.nn.batch_normalization(__tmp, 
            mean=__mean, variance=__variance, 
            offset=beta, scale=gamma, 
            variance_epsilon=1e-3
            )
            return bias + ACTIVATION(__tmp) if activation_flag else bias + (__tmp) 
            # for the last one, where there is no activation 
        
        def __get_var_from_weights_dict(prefix, strnum):
            # strnum is '_1' or ''
            kernal_var = weights[prefix + '/conv3d_transpose'+ strnum +'/kernel:0']
            # del weights[prefix + '/conv3d_transpose'+ strnum +'/kernel:0']
            bias_var = weights[prefix + '/conv3d_transpose'+ strnum +'/bias:0']
            # del weights[prefix + '/conv3d_transpose'+ strnum +'/bias:0']
            gamma_var = weights[prefix + '/batch_normalization'+ strnum +'/gamma:0']
            # del weights[prefix + '/batch_normalization'+ strnum +'/gamma:0']
            beta_var = weights[prefix + '/batch_normalization'+ strnum +'/beta:0']
            # del weights[prefix + '/batch_normalization'+ strnum +'/beta:0']
            # print("loading:", kernal_var, bias_var, gamma_var, beta_var)
            return (kernal_var, bias_var, gamma_var, beta_var)
            

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h_ = tensor_in
            h_ = tf.expand_dims(tf.expand_dims(tf.expand_dims(h_, 1), 1), 1)
            
            h = h_
            # print("tf.shape(h0)", h)
            # tf.shape(h0) Tensor("Model_2/Generator/ExpandDims_2:0", shape=(64, 1, 1, 1, 128), dtype=float32)


            # Shared network
            with tf.variable_scope('shared') as cur_scope:
                # tf.shape(h411) Tensor("Model_2/Generator/shared/Relu:0", shape=(64, 4, 1, 1, 512), dtype=float32)
                # h = tconv_layer(h, 512, (4, 1, 1), (4, 1, 1))        # 4, 1, 1
                # print("tf.shape(h411)", h)
                # 4, 4, 3
                h_ = nn_tconv_layer(h_,
                __get_var_from_weights_dict(cur_scope.name, ''), 
                (self.batch_size, 4, 1, 1, 512), 
                (4, 1, 1))        # 4, 1, 1

                
                # h = tconv_layer(h, 256, (1, 4, 3), (1, 4, 3))        # 4, 4, 3
                # print("tf.shape(h443)", h)
                # tf.shape(h443) Tensor("Model_2/Generator/shared/conv3d_transpose_2/Reshape_1:0", shape=(64, 4, 4, 3, 256), dtype=float32)                
                h_ = nn_tconv_layer(h_,
                __get_var_from_weights_dict(cur_scope.name, '_1'),
                (self.batch_size, 4, 4, 3, 256), 
                (1, 4, 3))        # 4, 4, 3

                # h = tconv_layer(h, 128, (1, 4, 3), (1, 4, 2))        # 4, 16, 7
                # print("tf.shape(h4167)", h)
                # tf.shape(h4167) Tensor("Model_2/Generator/shared/conv3d_transpose_3/Reshape_1:0", shape=(64, 4, 16, 7, 128), dtype=float32)

                h_ = nn_tconv_layer(h_,
                __get_var_from_weights_dict(cur_scope.name, '_2'),
                (self.batch_size, 4, 16, 7, 128), 
                # (1, 4, 2))        # 4, 16, 7
                (1, 4, 3))        # 4, 16, 7
                """ Very Strange Here?! Why 1,4,2?! A typo in the origin paper? """


            # Pitch-time private network
            with tf.variable_scope('pitch_time_private') as cur_scope:
                # s1 = [tconv_layer(h, 32, (1, 1, 12), (1, 1, 12))     # 4, 16, 84
                #       for _ in range(self.n_tracks)]
                # print("s1[0]",s1[0])
                # s1[0] Tensor("Model_2/Generator/pitch_time_private/conv3d_transpose/Reshape_1:0", shape=(64, 4, 16, 84, 32), dtype=float32)
                s1_ = []
                for i in range(self.n_tracks):
                    num_naming_str = '' if i == 0 else '_'+str(i)
                    s1_.append(nn_tconv_layer(h_,
                    __get_var_from_weights_dict(cur_scope.name, num_naming_str),
                    (self.batch_size, 4, 16, 84, 32 ), 
                    (1, 1, 12)) )     # 4, 16, 84
                   


                # s1 = [tconv_layer(s1[i], 16, (1, 3, 1), (1, 3, 1))   # 4, 48, 84
                #       for i in range(self.n_tracks)]
                # for i in range(self.n_tracks):
                #     print("i", i, "s1", s1[i])
                #     # i 0 s1 Tensor("Model_2/Generator/pitch_time_private/conv3d_transpose_5/Reshape_1:0", shape=(64, 4, 48, 84, 16), dtype=float32)
                s1_old_ = s1_ 
                s1_ = []
                for i in range(self.n_tracks, 2*self.n_tracks):
                    num_naming_str = '' if i == 0 else '_'+str(i)
                    s1_.append(nn_tconv_layer(s1_old_[i-self.n_tracks], 
                    __get_var_from_weights_dict(cur_scope.name, num_naming_str),
                    (self.batch_size, 4, 48, 84, 16), 
                    (1, 3, 1)) )   # 4, 48, 84        
                    

            # Time-pitch private network
            with tf.variable_scope('time_pitch_private') as cur_scope:
                # s2 = [tconv_layer(h, 32, (1, 3, 1), (1, 3, 1))       # 4, 48, 7
                #       for _ in range(self.n_tracks)]
                # print("s2[0]",s2[0])
                # s2[0] Tensor("Model_2/Generator/time_pitch_private/Relu:0", shape=(64, 4, 48, 7, 32), dtype=float32)
                s2_ = []
                for i in range(self.n_tracks):
                    num_naming_str = '' if i == 0 else '_'+str(i)
                    s2_.append(nn_tconv_layer(h_,
                    __get_var_from_weights_dict(cur_scope.name, num_naming_str),
                    (self.batch_size, 4, 48, 7, 32 ), 
                    (1, 3, 1)) )       # 4, 48, 7

                # s2 = [tconv_layer(s2[i], 16, (1, 1, 12), (1, 1, 12)) # 4, 48, 84
                #       for i in range(self.n_tracks)]
                # for i in range(self.n_tracks):
                #     print("i", i, "s2", s2[i])
                #     # i 0 s2 Tensor("Model_2/Generator/time_pitch_private/conv3d_transpose_5/Reshape_1:0", shape=(64, 4, 48, 84, 16), dtype=float32)

                s2_old_ = s2_ 
                s2_ = []
                for i in range(self.n_tracks, 2*self.n_tracks):
                    num_naming_str = '' if i == 0 else '_'+str(i)
                    s2_.append(nn_tconv_layer(s2_old_[i-self.n_tracks], 
                    __get_var_from_weights_dict(cur_scope.name, num_naming_str),
                    (self.batch_size, 4, 48, 84, 16), 
                    (1, 1, 12)) ) # 4, 48, 84  
                
            # h = [tf.concat((s1[i], s2[i]), -1) for i in range(self.n_tracks)]
            h_ = [tf.concat((s1_[i], s2_[i]), -1) for i in range(self.n_tracks)]

            # Merged private network
            with tf.variable_scope('merged_private') as cur_scope:
                # h = [norm(tconv3d(h[i], 1, (1, 1, 1), (1, 1, 1)))    # 4, 48, 84
                #      for i in range(self.n_tracks)]
                # print("h[0]",h[0])
                # h[0] Tensor("Model_2/Generator/merged_private/batch_normalization/batchnorm/add_1:0", shape=(64, 4, 48, 84, 1), dtype=float32)
                
                h_old_ = h_
                h_ = []
                for i in range(self.n_tracks):
                    num_naming_str = '' if i == 0 else '_'+str(i)
                    h_.append(nn_tconv_layer(h_old_[i],
                    __get_var_from_weights_dict(cur_scope.name, num_naming_str),
                    (self.batch_size, 4, 48, 84, 1), 
                    (1, 1, 1), activation_flag=False) )    # 4, 48, 84
                
                # h = tf.concat(h, -1)
                h_ = tf.concat(h_, -1)
                
        # return tanh(h)
        return tanh(h_)
