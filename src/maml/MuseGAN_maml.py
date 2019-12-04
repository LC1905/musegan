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
        self.num_discriminator_updates = 3
        self.num_generator_updates = 1

        """ must call construct_model() after initializing MAML! """
        
        self.classification = False
        self.test_num_updates = test_num_updates

        self.meta_batch_size = 1 

        self.stop_grad = True  # 'if True, do not use second derivatives in meta-optimization (for speed)'


    #############################################################################################
    #TODO: replace with MuseGAN's architecture; note that MuseGAN load the network from YAML file.

    ##############################################################################################

    def construct_model(self, weights, input_tensors=None, prefix='metatrain_'): 
        

        # a: training data for inner gradient, b: test data for meta gradient

        #############################################################################################
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
            self.weights = weights
            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []

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
                #inputa = tf.expand_dims(inputa, 0)
                #inputb = tf.expand_dims(inputb, 0)
                # task_outputbs, task_lossesb = [], []
                
                generator_val_losses = []
                discriminator_val_losses = []
                dis_first = True
                gen_first = True
                # if self.classification:
                    # task_accuraciesb = []

                weight_keys = weights.keys()
                weight_values = list(weights.values())
                # ================ fast gradient descent discriminator() ===================== #
                discriminator_fast_weights = dict(zip(weight_keys, weight_values))
                discriminator_fast_weights_val = dict(zip(weight_keys, weight_values))
                for _ in range(self.num_discriminator_updates):
                    # Update on train.
                    nodes = self.inner_model.get_generator_loss_with_weights_input(inputa, discriminator_fast_weights, config=self.config, params=self.params)
                    gd_loss = nodes['dis_loss']
                    if dis_first:
                        discriminator_train_loss = nodes['dis_loss']
                    dis_first = False

                    grads = tf.gradients(gd_loss, list(discriminator_fast_weights.values()))
                    if self.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    weight_gradients = dict(zip(weight_keys, grads))
                   
                    # Fast update weights dictionary
                    for key in weight_keys:
                        if 'Model/Discriminator/' in key:
                            discriminator_fast_weights[key] -= self.update_lr * weight_gradients[key]
 
                    # Get validation loss, which will be used to meta-train discriminator.
                    nodes = self.inner_model.get_generator_loss_with_weights_input(inputb, discriminator_fast_weights, config=self.config, params=self.params)
                    discriminator_val_losses.append(nodes['dis_loss'])

                    # Update on val.
                    nodes = self.inner_model.get_generator_loss_with_weights_input(inputb, discriminator_fast_weights_val, config=self.config, params=self.params)
                    gd_loss = nodes['dis_loss']
                    # with tf.Session() as sess:
                        # print('=== GD Loss {} ==='.format(gd_loss))
                    grads = tf.gradients(gd_loss, list(discriminator_fast_weights_val.values()))
                    if self.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                        weight_gradients = dict(zip(weight_keys, grads))
                        for key in weight_keys:
                            if 'Model/Discriminator/' in key:
                                discriminator_fast_weights_val[key] -= self.update_lr * weight_gradients[key]

                # =================================================================================
                # ==== fast gradient descent generator with dis.fast weights on train =============== #
                
                # Learn with train-updated discriminator; evaluate with val-updated discriminator.
                generator_fast_weights_train = dict()
                generator_fast_weights_val = dict()
                for key in weight_keys:
                    if 'Model/Generator/' in key:
                        generator_fast_weights_train[key] = weights[key]
                        generator_fast_weights_val[key] = weights[key]
                    elif 'Model/Discriminator/' in key:
                        generator_fast_weights_train[key] = discriminator_fast_weights[key]
                        generator_fast_weights_val[key] = discriminator_fast_weights_val[key]
                    else:
                        raise NameError(key)
                
                for _ in range(self.num_generator_updates):
                    # Fast update.
                    nodes = self.inner_model.get_generator_loss_with_weights_input(inputa, generator_fast_weights_train, config=self.config, params=self.params)
                    gd_loss = nodes['gen_loss']
                    if gen_first:
                        generator_train_loss = nodes['gen_loss'] 
                    gen_first = False
                    grads = tf.gradients(gd_loss, list(generator_fast_weights_train.values()))
                    if self.stop_grad:
                        grad = [tf.stop_gradient(grad) for grad in grads]
                    weight_gradients = dict(zip(weight_keys, grads))

                    # Fast update weights dictionary for generator:
                    for key in weight_keys:
                        if 'Model/Generator/' in key:
                            generator_fast_weights_train[key] -= self.update_lr * weight_gradients[key]
                            generator_fast_weights_val[key] = generator_fast_weights_train[key]

                    # Evaluate on val set.
                    nodes = self.inner_model.get_generator_loss_with_weights_input(inputb, generator_fast_weights_val, config=self.config, params=self.params)
                    generator_val_losses.append(nodes['gen_loss'])
                # ================================================================================ #
               
                task_output = [generator_train_loss, discriminator_train_loss, generator_val_losses, discriminator_val_losses]
                """
                # ============ fast gradient descent discriminator on val ======================== #
                discriminator_fast_weights = dict(zip(weight_keys, weight_values))
                for _ in range(self.num_discriminator_updates):
                    nodes = self.inner_model.get_discriminator_loss_with_weights_input(inputb, 
                        discriminator_fast_weights, config=self.config, params=self.params)
                    gd_loss = nodes['dis_loss']
                    grads = tf.gradients(gd_loss, list(discriminator_fast_weights.values()))
                    if self.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    weight_gradients = dict(zip(weight_keys, grads))
                    for key in weight_keys:
                        if 'Model/Discriminator/' in key:
                            discriminator_fast_weights[key] -= self.update_lr * weight_gradients[key]

                # ================================================================================ #
                # ========== compute generator validation loss =================================== #
                # Merge generator's updated weight with discriminator's updated weight (on val)
                merged_fast_weights = dict()
                for key in weight_keys:
                    if 'Model/Discriminator/' in key:
                        merged_fast_weights[key] = discriminator_fast_weights[key]
                    elif 'Model/Generator/' in key:
                        merged_fast_weights[key] = generator_fast_weights[key]
                    else:
                        raise NameError(key)
                # Only update generator for 1 step now.
                # If multiple updates, we need train-updated discriminator for learning; and val-updated discriminator for validation.
                nodes = self.inner_model.get_generator_loss_with_weights_input(inputb, merged_fast_weights,
                    config=self.config, params=self.params)
                    
                generator_val_losses.append(nodes['gen_loss'])
                   
                """
                """
               # AHA: before num_update-times meta-learning.
                #########################################################################################
                # task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                # task_lossa = self.loss_func(task_outputa, labela)
                
                nodes = self.inner_model.get_generator_loss_with_weights_input(inputa, weights, 
                config=self.config, params=self.params)
                
                # return [nodes['gen_loss'], [nodes['gen_loss'] for _ in range(5)]]
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
                
                """
                """
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

                """
                """
                # A line by line translation
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
                """

                return task_output

            # if FLAGS.norm is not 'None':
            #     # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
            #     unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, tf.float32, [tf.float32] * self.num_generator_updates, self.num_discriminator_updates]
            """
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            """
            # result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            # print("feeding into map_fn, inputa:", self.inputa)
            
            
            
            # result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=1)
            
            generator_train_loss, discriminator_train_loss, generator_val_losses,\
                discriminator_val_losses  = task_metalearn((self.inputa, self.inputb, self.labela, self.labelb))
            
            """
            if self.classification:# False 
                # outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
                lossesa, lossesb, accuraciesa, accuraciesb = result
            else:
                # lossesa, lossesb  = result
                generator_train_loss, discriminator_train_loss, generator_val_losses, discriminator_val_losses = result
            """
        ## Performance & Optimization
        if 'train' in prefix:
            # self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
            # self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            # self.outputas, self.outputbs = outputas, outputbs
            """
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(self.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
            """

            self.generator_loss_1 = generator_loss_1 = tf.reduce_sum(generator_train_loss) / tf.to_float(self.meta_batch_size)
            self.generator_losses_2 = generator_losses_2 = [tf.reduce_sum(generator_val_losses[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_generator_updates)] 
            self.discriminator_loss_1 = discriminator_loss_1 = tf.reduce_sum(discriminator_train_loss) / tf.to_float(self.meta_batch_size)
            self.discriminator_losses_2 = discriminator_losses_2 = [tf.reduce_sum(discriminator_val_losses[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_discriminator_updates)] 
            
            if self.metatrain_iterations > 0:
                gen_optimizer = tf.train.AdamOptimizer(self.meta_lr)
                dis_optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gen_gvs = gen_gvs = gen_optimizer.compute_gradients(self.generator_losses_2[self.num_generator_updates-1])
                self.dis_gvs = dis_gvs = dis_optimizer.compute_gradients(self.discriminator_losses_2[self.num_discriminator_updates-1])
                # if self.datasource == 'miniimagenet':
                    # gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.gen_metatrain_op = gen_optimizer.apply_gradients(gen_gvs)
                self.dis_metatrain_op = dis_optimizer.apply_gradients(dis_gvs)

        else:
            self.metaval_generator_loss_1 = generator_loss_1 = tf.reduce_sum(generator_train_loss) / tf.to_float(self.meta_batch_size)
            self.metaval_generator_losses_2 = generator_losses_2 = [tf.reduce_sum(generator_val_losses[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_generator_updates)] 
            self.metaval_discriminator_loss_1 = discriminator_loss_1 = tf.reduce_sum(discriminator_train_loss) / tf.to_float(self.meta_batch_size)
            self.metaval_discriminator_losses_2 = discriminator_losses_2 = [tf.reduce_sum(discriminator_val_losses[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_discriminator_updates)]
           
            """
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
            
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(self.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
            """
        ## Summaries
        tf.summary.scalar(prefix+'Pre-update generator loss', generator_loss_1)
        tf.summary.scalar(prefix+'Pre-update discriminator loss', discriminator_loss_1)
        # if self.classification:
            # tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(self.num_generator_updates):
            tf.summary.scalar(prefix+'Post-update generator loss, step ' + str(j+1), generator_losses_2[j])
        for j in range(self.num_discriminator_updates):
            tf.summary.scalar(prefix+'Post-update discriminator loss, step ' + str(j+1), discriminator_losses_2[j])
            # if self.classification:
                # tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])
