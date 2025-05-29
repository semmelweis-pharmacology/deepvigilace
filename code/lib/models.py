#%% Import libraries

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Reshape, Dot, Concatenate, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model


#%% Define functions for model building

def build_ntx(target_dict, context_dict, embedding_dim):
    
    input_target = Input((1), name = 'target_input')
    input_context = Input((1), name = 'context_input')
    
    # For older Tensorflow versions, the following commented section might be
    # needed for the transformation of the input integers into one-hot vectors.
    
    # target = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
    #     max_tokens = len(target_dict), output_mode = "binary") (input_target)
    # context = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
    #     max_tokens = len(context_dict), output_mode = "binary") (input_context)
    
    target = tf.keras.layers.CategoryEncoding(
    num_tokens = len(target_dict), output_mode = 'one_hot', sparse = False) (input_target)
    context = tf.keras.layers.CategoryEncoding(
    num_tokens = len(context_dict), output_mode = 'one_hot', sparse = False) (input_context)
    
    target_embedding = Dense(units = embedding_dim,
                              activation = None,
                              use_bias = True,
                              kernel_initializer = "glorot_uniform",
                              bias_initializer = "zeros",
                              name = 'target_embedding')
    
    context_embedding = Dense(units = embedding_dim,
                              activation = None,
                              use_bias = True,
                              kernel_initializer = "glorot_uniform",
                              bias_initializer = "zeros",
                              name = 'context_embedding')
    
    target_embedding_out = Reshape(target_shape = (1, embedding_dim))(target_embedding(target))
    context_embedding_out = Reshape(target_shape = (1, embedding_dim))(context_embedding(context))
    
    projector_head = Dense(units = 128,
                           activation = 'relu')
    projector_head_2 = Dense(units = 128,
                             activation = None)
    
    # Propagate the embedding vectors through the same projector heads
    target_head = projector_head_2 (projector_head (target_embedding_out))
    context_head = projector_head_2 (projector_head (context_embedding_out))
    
    # Concatenate the two vectors as output for the calculation of the contrastive loss later
    concat = tf.concat([target_head, context_head], 1)
    model = Model([input_target, input_context], concat)
    
    return model


def build_nsg(target_dict, context_dict, embedding_dim, norm_vecs):
    
    input_target = Input((1), name = 'target_input')
    input_context = Input((1), name = 'context_input')

    # For older Tensorflow versions, the following commented section might be
    # needed for the transformation of the input integers into one-hot vectors.

    # target = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
    #     max_tokens=len(target_dict), output_mode = "binary") (input_target)
    # context = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
    #     max_tokens=len(context_dict), output_mode = "binary") (input_context)
    
    target = tf.keras.layers.CategoryEncoding(
    num_tokens = len(target_dict), output_mode = 'one_hot', sparse = False) (input_target)
    context = tf.keras.layers.CategoryEncoding(
    num_tokens = len(context_dict), output_mode = 'one_hot', sparse = False) (input_context)
    
    target_embedding = Dense(units = embedding_dim,
                              activation = None,
                              use_bias = True,
                              kernel_initializer = "glorot_uniform",
                              bias_initializer = "zeros",
                              name = 'target_embedding')
    
    context_embedding = Dense(units = embedding_dim,
                              activation = None,
                              use_bias = True,
                              kernel_initializer = "glorot_uniform",
                              bias_initializer = "zeros",
                              name = 'context_embedding')
    
    # Calculate the dot product (or cosine if norm_vecs = True) of the embedding vectors then apply non-linearity 
    dot_product = Dot(axes = 1, normalize = norm_vecs)([target_embedding(target), context_embedding(context)])
    output = Dense(1, activation='sigmoid', kernel_initializer = "glorot_uniform")(dot_product)
    model = Model([input_target, input_context], output)
    
    return model


def build_classifier(chem_input_dim, bio_input_dim, aer_input_dim):
        
    chem_input_layer = Input(shape = chem_input_dim)    
    chem_hidden = Dense(aer_input_dim/2, activation = None,
                          use_bias = True,
                          kernel_initializer = "glorot_uniform",
                          bias_initializer = "zeros") (chem_input_layer)
    
    bio_input_layer = Input(shape = bio_input_dim)
    bio_hidden = Dense(aer_input_dim/2, activation = None,
                          use_bias = True,
                          kernel_initializer = "glorot_uniform",
                          bias_initializer = "zeros") (bio_input_layer)
    
    aer_input_layer = Input(shape = aer_input_dim)
    
    hidden = Concatenate(axis = 1)([chem_hidden, bio_hidden, aer_input_layer])
    hidden = BatchNormalization()(hidden)
    
    hidden = Dense(1024, activation = None)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = LeakyReLU()(hidden)
    hidden = Dropout(0.4)(hidden)
    
    hidden = Dense(256, activation = None)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = LeakyReLU()(hidden)
    
    out_layer = Dense(1, activation = 'sigmoid')(hidden)

    model = Model([chem_input_layer, bio_input_layer, aer_input_layer], out_layer)
    
    return model


def build_classifier_without_emb(chem_input_dim, bio_input_dim, aer_input_dim, aer_num):
        
    chem_input_layer = Input(shape = chem_input_dim)    
    chem_hidden = Dense(aer_input_dim/2, activation = None,
                          use_bias = True,
                          kernel_initializer = "glorot_uniform",
                          bias_initializer = "zeros") (chem_input_layer)
    
    bio_input_layer = Input(shape = bio_input_dim)
    bio_hidden = Dense(aer_input_dim/2, activation = None,
                          use_bias = True,
                          kernel_initializer = "glorot_uniform",
                          bias_initializer = "zeros") (bio_input_layer)
    
    aer_input_layer = Input(shape = 1)
    
    # For older Tensorflow versions, the following commented section might be
    # needed for the transformation of the input integers into one-hot vectors.
    
    # aer_embedding = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
    #     max_tokens = aer_num, output_mode = "binary") (aer_input_layer)
    
    aer_embedding =  tf.keras.layers.CategoryEncoding(
        num_tokens = aer_num, output_mode = 'one_hot', sparse = False) (aer_input_layer)
    
    aer_hidden = Dense(aer_input_dim, activation = None,
                          use_bias = True,
                          kernel_initializer = "glorot_uniform",
                          bias_initializer = "zeros") (aer_embedding)

    hidden = Concatenate(axis = 1)([chem_hidden, bio_hidden, aer_hidden])
    hidden = BatchNormalization()(hidden)
    
    hidden = Dense(1024, activation = None)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = LeakyReLU()(hidden)
    hidden = Dropout(0.4)(hidden)
    
    hidden = Dense(256, activation = None)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = LeakyReLU()(hidden)
    
    out_layer = Dense(1, activation = 'sigmoid')(hidden)

    model = Model([chem_input_layer, bio_input_layer, aer_input_layer], out_layer)
    
    return model


#%% Define functions for loss calculation

def calc_ntx_loss(norm_vecs, temperature):
    
    @tf.autograph.experimental.do_not_convert
    def contrastive_loss(y_true, output):
        
        # Prepare the embedding vectors
        (emb_1, emb_2) = tf.split(output, num_or_size_splits = 2, axis = 1)
        batch_size = tf.shape(emb_1)[0]
        emb_1 = tf.reshape(emb_1, [batch_size, -1])
        emb_2 = tf.reshape(emb_2, [batch_size, -1])
        
        # Normalize the embedding vectors if needed, resulting in cosine similarity
        # or dot product otherwise
        if norm_vecs == True:
            emb_1 = tf.math.l2_normalize(emb_1, axis = 1)
            emb_2 = tf.math.l2_normalize(emb_2, axis = 1)
        
        # Calculate the cosine similarity (or dot product)
        # and scale it with the temperature parameter
        sim_matrix = (tf.matmul(emb_1, emb_2, transpose_b = True) / temperature)
        # Use the index of words in the batch as their assigned labels
        # so that for the multiclass classification training can be achieved
        # by using the identity matrix as the expected outcome. This will enforce
        # the model to create embedding vectors such that the cosine similarity
        # of a given target word is highest with its own corresponding context
        # word and close to 0 with all the other ones. As cosine similarity goes
        # from -1 to +1, we need from_logits = True in the cross entropy calculation
        # which first applies softmax to normalize the values.
        contrastive_labels = tf.range(batch_size)
        
        # Calculate cross entropy loss from both direction (target-context)
        # Then average it (div by 2)
        loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
                contrastive_labels, sim_matrix, from_logits = True)
        loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
                contrastive_labels, tf.transpose(sim_matrix), from_logits = True)
        
        return (loss_1_2 + loss_2_1) / 2
    
    return contrastive_loss
            
    