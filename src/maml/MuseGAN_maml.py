""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import maml.special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

# from tensorflow.python.platform import flags
from maml.utils import mse, xent, conv_block, normalize

# FLAGS = flags.FLAGS

class MAML:
    def __init__(self, test_num_updates=5, # dim_input=1, dim_output=1, 
    inner_model=None, config=None, params=None):
        
        self.inner_model = inner_model # reference to the true model 
        self.config = config
        self.params = params # The global config and params for the inner model. 
        self.update_lr = 1e-3 
        self.meta_lr = 0.001 # flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
        self.metatrain_iterations = 70000
        self.datasource = ''
        self.num_updates = 5
        """ must call construct_model() after initializing MAML! """
        
        self.classification = False
        self.test_num_updates = test_num_updates

        self.meta_batch_size = 32 

        self.stop_grad = True  # 'if True, do not use second derivatives in meta-optimization (for speed)'


    #############################################################################################
    #TODO: replace with MuseGAN's architecture; note that MuseGAN load the network from YAML file.

    ##############################################################################################

    def construct_model(self, weights, input_tensors=None, prefix='metatrain_'): 
        

        # a: training data for inner gradient, b: test data for meta gradient

        #############################################################################################
        #TODO: In MuseGAN, we inputa and inputb should just be noise; while labela and labelb are real data (music in the given category)
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']
        ##############################################################################################

        
        # with tf.variable_scope('model', reuse=None) as training_scope:
        for _ in range(1):# dummy, only for creating indentation
            
            # if 'weights' in dir(self):
            #     training_scope.reuse_variables()
            #     weights = self.weights
            # else:
            #     # Define the weights
            #     self.weights = weights = self.construct_weights()
            self.weights = weights

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            
            ####################
            #TODO: Man I really don't like the way they name things..
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            ####################

            # num_updates = max(self.test_num_updates, FLAGS.num_updates)
            num_updates = self.test_num_updates 
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                # This sounds like a main train function ? merge with musegan train main 
                inputa, inputb, labela, labelb = inp
                # IMPORTANT! MUST unsqueeze inputa, inputb! 
                inputa = tf.expand_dims(inputa, 0)
                inputb = tf.expand_dims(inputb, 0)
                task_outputbs, task_lossesb = [], []

                if self.classification:
                    task_accuraciesb = []

                # AHA: before num_update-times meta-learning.
                #########################################################################################
                #TODO: task_outputa will be our fake music; and loss will be BCELoss between the fake and real in the given class. Probably nothing needs to be changed here if the architecture is well-defined.
                # Also, what is "reuse" doing here?
                
                # task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                # task_lossa = self.loss_func(task_outputa, labela)
                # print("inside task_metalearn, we get input a:", inputa)
                nodes = self.inner_model.get_generator_loss_with_weights_input(inputa, weights, 
                config=self.config, params=self.params)
                task_lossa = nodes['gen_loss']
                # Future TODO: use a better update policy. considering the tricks inside the origin GAN.

                #########################################################################################

                grads = tf.gradients(task_lossa, list(weights.values()))
                if self.stop_grad:
                    # Note: flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
    
                #########################################################################################
                #TODO: only update the weights of the generator.
                """ Yes! Only the generator! """
                manual_updated_weigts_list = [
                    weights[key] - self.update_lr*gradients[key] if 'Model/Generator/' in key \
                        else weights[key] # NOT updating the discriminator, whose prefix is 'Model/Discriminator/'
                    for key in weights.keys() 
                ] 
                
                fast_weights = dict(zip(weights.keys(), manual_updated_weigts_list)) # AHA: manual gradient descent on G_{\theta}
                
                # output = self.forward(inputb, fast_weights, reuse=True)
                # task_outputbs.append(output)
                # task_lossesb.append(self.loss_func(output, labelb))
                nodes = self.inner_model.get_generator_loss_with_weights_input(inputb, fast_weights, config=self.config, params=self.params)
                task_lossesb.append(nodes['gen_loss'])
                

                #########################################################################################

                
                """ A line by line translation here! """
                for j in range(num_updates - 1):
                    # loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    nodes = self.inner_model.get_generator_loss_with_weights_input(inputa, fast_weights, config=self.config, params=self.params)
                    loss = nodes['gen_loss']
                
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if self.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))

                    # fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    manual_updated_weigts_list = [
                        fast_weights[key] - self.update_lr*gradients[key] if 'Model/Generator/' in key \
                            else fast_weights[key] # NOT updating the discriminator, whose prefix is 'Model/Discriminator/'
                        for key in fast_weights.keys() 
                    ]
                    fast_weights = dict(zip(fast_weights.keys(), manual_updated_weigts_list)) # AHA: manual gradient descent on G_{\theta}

                    # output = self.forward(inputb, fast_weights, reuse=True)
                    # task_outputbs.append(output)
                    # task_lossesb.append(self.loss_func(output, labelb))                    
                    nodes = self.inner_model.get_generator_loss_with_weights_input(inputb, fast_weights, config=self.config, params=self.params)
                    # task_lossb = nodes['gen_loss']
                    task_lossesb.append(nodes['gen_loss'])  


                task_outputa = None
                task_outputbs = None 
                # task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                task_output = [task_lossa, task_lossesb] 

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                    for j in range(num_updates):
                        task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            # if FLAGS.norm is not 'None':
            #     # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
            #     unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            # out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            out_dtype = [tf.float32, [tf.float32]*num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            # result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            # print("feeding into map_fn, inputa:", self.inputa)
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=self.meta_batch_size)
            if self.classification:# False 
                # outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
                lossesa, lossesb, accuraciesa, accuraciesb = result
            else:
                lossesa, lossesb  = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            # self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(self.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)



            if self.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[self.num_updates-1])
                if self.datasource == 'miniimagenet':
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(self.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])
