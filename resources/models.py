import numpy as np
import pandas as pd
# import tqdm
import tensorflow as tf 
from tensorflow import keras
import tensorflow_probability as tfp 
import resources.data_loader as dl

#---------------------------------------------Preprocess Sequence Embeddings---------------------------------------
# Atchley Factor Vectorization 
def get_seq_vec_shape(seq_vectors):
    """Return model depth"""

    seq_vec_shape = seq_vectors.shape[1]

    return seq_vec_shape

class AtchleyFactorVectorizer(tf.keras.layers.Layer):
    """An Embedding Lookup algorithm"""
    def __init__(self, seq_vectors):
        super(AtchleyFactorVectorizer, self).__init__()
        self.seq_vectors = tf.constant(seq_vectors, dtype=tf.float32)

    def call(self, inputs):
        embedded_inputs = tf.nn.embedding_lookup(self.seq_vectors, inputs)
        return embedded_inputs

# Positional Encoder
def encode_position(
        seq_vec_shape: int, 
        max_seq_length: int,
):
    """Returns the positional embedding vector"""

    initial_positions = np.arange(max_seq_length)[:, np.newaxis]
    positions = np.repeat(initial_positions, seq_vec_shape, axis=1)

    angle_rates = 1/1000
    angle_rads = positions * angle_rates

    s = np.sin(angle_rads)[::2]
    c = 1 - np.cos(angle_rads)[1::2]

    stacked_sc = np.vstack([s, c])
    encoded_positions = tf.cast(stacked_sc, dtype=tf.float32)

    return encoded_positions


# Build a Positional Encoding Layer using Keras Subclassing API
class PositionalEncodingLayer(tf.keras.layers.Layer):
    """Vectorize the raw sequences using Atchley Factors and add the positional embedding vector on top of the embedding vector"""
    
    def __init__(self, seq_vectors, seq_vec_shape, max_seq_length):
        super().__init__()
        self.seq_vec_shape = seq_vec_shape
        self.max_seq_length = max_seq_length
        self.embedding = AtchleyFactorVectorizer(seq_vectors)
        self.encode_position = encode_position(
            seq_vec_shape=seq_vec_shape, 
            max_seq_length=max_seq_length
        )
        
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.seq_vec_shape, tf.float32))
        x = x + self.encode_position[tf.newaxis, :self.max_seq_length, :]

        return x
 
def position_encode_data(
        input_df: any,
        token_table: any,
        seq_type: str,
        seq_vectors: any,
        max_seq_length: int,
        seq_vec_shape: int,
):
    """Loop over all the sequences in the dataset:
    1. Perform Atchley Factor vectorization
    2. Apply Positional Encoding to the vectorized sequences
    """

    # Read data and get the sample size
    sample_size = input_df.shape[0]

    # Initialize the Positional Encoding Layer
    positinal_encoder = PositionalEncodingLayer(seq_vectors=seq_vectors, 
                                                seq_vec_shape=seq_vec_shape, 
                                                max_seq_length=max_seq_length)

    # Initialize the embeddings
    tokens = np.zeros([sample_size, max_seq_length], dtype='int')
    embs = np.zeros([sample_size, max_seq_length, seq_vec_shape])

    # Encode each sequence in the sample
    for i in range(sample_size):
        tokens[i] = dl.map_idx_to_token(token_table=token_table,
                                        input_sequence = input_df[seq_type].iloc[i], 
                                        max_seq_length = max_seq_length)
        embs[i] = positinal_encoder(tokens[i])

    # Reshape to feed as inputs to the models
    embs = embs.reshape(-1, max_seq_length*seq_vec_shape)

    return embs

# Autoencoder
def get_ae_input_shape(
        max_seq_length: int,
        seq_vec_shape: int,        
):
    """Get input shape of the Autoencoder model"""
    
    ae_input_shape = max_seq_length * seq_vec_shape
    
    return ae_input_shape

# Building an autoencoder model using Keras Subclassing API
# @keras.saving.register_keras_serializable()
# class AutoEncoder(tf.keras.models.Model):
#     """Define an autoencoder with:
#     1. Encoder: compresses the vectorized (& positional embedded) sequence inputs into a latent vector
#     2. Decoder: reconstructs  the original inputs from the latent space"""

#     def __init__(self, latent_dim, ae_input_shape):
#         super(AutoEncoder, self).__init__()

#         # Initialize model parameters
#         self.latent_dim = latent_dim
#         self.ae_input_shape = ae_input_shape

#         # Encoder
#         self.encoder = keras.Sequential([
#             keras.Input(shape=self.ae_input_shape),
#             keras.layers.Normalization(axis = None),
#             keras.layers.Dense(units=256, activation='relu'),
#             keras.layers.BatchNormalization(),
#             keras.layers.Reshape((16,16,1)),
#             keras.layers.Conv2D(filters=64, kernel_size=2, strides=(1,1), activation='relu'),
#             keras.layers.BatchNormalization(),
#             keras.layers.Flatten(),
#             keras.layers.Dense(units=512, activation='relu'),
#             keras.layers.Dense(units=self.latent_dim, name='latent')
#         ])
        
#         # Decoder
#         self.decoder = keras.Sequential([
#             keras.layers.Dense(units=512),
#             keras.layers.Dense(units=12544, activation='relu'),
#             keras.layers.Reshape((14,14,64)),
#             keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=1, activation='relu'),
#             keras.layers.BatchNormalization(),
#             keras.layers.Flatten(),
#             keras.layers.Dense(units=256),
#             keras.layers.Dense(units=self.ae_input_shape)
#         ])

#     def call(self, inputs, training=False):
#         encoded = self.encoder(inputs)
#         decoded = self.decoder(encoded)
#         return decoded
    
# Building an autoencoder model using Keras Functional API
def create_autoencoder(
        latent_dim: int,
        ae_input_shape: int,
):
    """Define an autoencoder with:
    1. Encoder: compresses the vectorized (& positional embedded) sequence inputs into a latent vector
    2. Decoder: reconstructs  the original inputs from the latent space"""
    
    # Encoder
    input = keras.Input(shape=ae_input_shape)
    x = keras.layers.Normalization(axis=None)(input)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Reshape((16,16,1))(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=2, strides=(1,1), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1,1), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    latent_space = keras.layers.Dense(latent_dim, name='latent')(x)

    # Decoder
    x = keras.layers.Dense(512)(latent_space)
    x = keras.layers.Dense(12544, activation='relu')(x)
    x = keras.layers.Reshape((14,14,64))(x)
    x = keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=1, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256)(x)
    decoded = keras.layers.Dense(ae_input_shape)(x)
    
    # Create the model
    model = keras.Model(inputs = input, outputs = decoded)

    return model

# Encoder Decoder Model
def get_ed_input_shape(dim: int):
    """Return the input shape of the Encoder Decoder model"""

    shape = (dim, dim, 1)

    return shape

# Build an Encoder Decoder Model using Functional API
def create_encoder_decoder(
        latent_dim: int,
        ed_input_shape: int,
):
    """Define an autoencoder with:
    1. Encoder: compresses the inputs into a latent space
    2. Decoder: takes the latent space and returns the decoded ouput"""

    # Encoder
    input = keras.Input(shape = ed_input_shape)
    # x = keras.layers.Normalization(axis=None)(input)
    x = keras.layers.Conv2D(32,(2,2))(input)
    x = keras.layers.Conv2D(filters=128, kernel_size=2, strides=1, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=2, strides=1, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=2, strides=1, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=16, kernel_size=2, strides=1, activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    encoded = keras.layers.Dense(latent_dim*9, name='latent')(x)

    # Decoder
    x = keras.layers.Dense(latent_dim*9)(encoded)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(784, activation='relu')(x)
    x = keras.layers.Reshape((7,7,16))(x)
    x = keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2,activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2,activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=2, strides=2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(latent_dim)(x)
    decoded = keras.layers.Reshape(ed_input_shape)(x)

    # Create the model
    model = keras.Model(inputs=input, outputs=decoded)

    return model    

# Train the model
def get_model(model):
    """Create an autoencoder model and return the model"""

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001), 
        loss='mse'
    )

    return model

def train_model(
        fpath_ckp,
        inputs,
        outputs,
        model
):
    """Create checkpoint for model weights, save the best model weights and return the model"""

    # Create early stopping and model checkpoint callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min', min_delta=0.0001),
        keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, monitor='loss',min_lr=1e-17, min_delta=0.5),
        keras.callbacks.ModelCheckpoint(fpath_ckp, save_weights_only=True)
    ]

    # Train the model
    history = model.fit(x=inputs, y=outputs,
        epochs=100,
        batch_size=256,
        shuffle=True,
        validation_split=0.2,
        callbacks=callbacks,
    )

    return model

def save_and_load_best_model(
        fpath_ckp: str,
        model: any,
):
    """Save and load the reconstructed model"""
    model.save_weights(fpath_ckp)
    model.load_weights(fpath_ckp)

    return model

def get_latent_space(model, input):
    """Get latent space of the autoencoder model"""

    # Instantiate a model to get the latent layer
    latent_model = keras.Model(
        inputs = model.input, 
        outputs = model.get_layer('latent').output
    )

    # Make prediction on the latent model
    latent_space = latent_model.predict(input)

    return latent_space

#---------------------------------------Prepare Categorical Embeddings---------------------------------
def get_cat_latent_dim(
    x: int,
):
    latent_dim = x**2

    return latent_dim

def get_one_hot_data(
    input_data: any,
    cols: any,
):
    one_hot_data = pd.get_dummies(input_data[cols], columns=cols, dtype='float')

    return one_hot_data

def get_cat_input_shape(
    one_hot_data: any
):
    ae_cat_input_shape = one_hot_data.shape[1]

    return ae_cat_input_shape

def create_cat_autoencoder(
    latent_dim,
    ae_cat_input_shape
):
    # Encoder
    input = keras.Input(shape=ae_cat_input_shape)
    x = keras.layers.Normalization(axis = None)(input)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    latent_space= keras.layers.Dense(75, activation='relu', name = "latent")(x)

    # Decoder
    x = keras.layers.Dense(latent_dim)(latent_space)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(256, activation='tanh')(x)
    decoded = keras.layers.Dense(ae_cat_input_shape)(x)
        
    # Create the model
    model = keras.Model(inputs = input, outputs = decoded)

    return model

def get_cat_autoencoder(model):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = 'mse'
    )

    return model

def train_cat_autoencoder(
    model,
    one_hot_data
):
    es_callback = keras.callbacks.EarlyStopping(
        monitor="loss", 
        patience=10, 
        restore_best_weights=True
        )
    reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.1, 
        patience=5, 
        monitor='loss',
        min_lr=1e-8, 
        min_delta=0.005
        )
    model.fit(
        one_hot_data, 
        one_hot_data, 
        batch_size=32,
        epochs=20,
        shuffle=True, 
        callbacks=[es_callback, reduce_learning_rate]
    )


#-------------------------------------------Build a classifier-----------------------------------------
def create_bayesian_classifier(
        train_input: any,
        num_class: int,
    ):
    """Create a Keras model to predict peptide binding"""

    # KL divergence weighted by the number of training examples
    kl_divergence_function = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (train_input.shape[0] * 1.0)
    
    # Define input and output 
    input = keras.Input(shape=train_input.shape[1:3])
    x = tf.keras.layers.Flatten()(input)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tfp.layers.DenseFlipout(16, activation='relu', kernel_divergence_fn=kl_divergence_function)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output = tfp.layers.DenseFlipout(units=num_class, activation='softmax', kernel_divergence_fn=kl_divergence_function)(x)
    
    # Create the model
    model = keras.Model(inputs = input, outputs = output)

    return model

def get_bayesian_classifier(model, learning_rate):

    model.compile(
        optimizer = keras.optimizers.legacy.Adamax(learning_rate=learning_rate),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        weighted_metrics=[
            tf.keras.metrics.AUC(from_logits=True, multi_label=False),
            tf.keras.metrics.CategoricalAccuracy()
        ]
    )

    return model

def train_bayesian_classifier(
        fpath_ckp,
        model,
        epochs,
        sample_weight, 
        batch_size, 
        train_input, 
        val_input, 
        train_binary_labels, 
        val_binary_labels
):

    # Create an early stopping function to stop training when auc stops improving
    es_callback = keras.callbacks.EarlyStopping(
        monitor='auc', 
        patience=20, 
        restore_best_weights=True
    )

    # Initialize tqdm callback with default parameters to show progress bars 
    # tqdm_callback = tfa.callbacks.TQDMProgressBar() 

    # Create a model checkpoint
    cp_callback = keras.callbacks.ModelCheckpoint(
        fpath_ckp,
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
    )

    # Train the model
    model.fit(x=train_input, y=train_binary_labels, 
              epochs=epochs,
              sample_weight=sample_weight,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(val_input, val_binary_labels),
              callbacks=[
                  es_callback,
                  cp_callback,
                  # tqdm_callback
              ],
              verbose=0,
    )

    return model

def tune_mira_train_model(
        train_input,
        val_input,
        train_binary_labels,
        val_binary_labels,
        y_train,
        class_ratio_map,
        model,
        epochs,
        learning_rates, 
        batch_sizes, 
        fpath_ckp
    ):
    """Tune the sample weight and save the best model."""
    best_auc = 0 # initial auc

    for minor in [-0.25,-0.5,0,0.25,0.5]:
        for major in [1,3,6,9]:
            sample_weight = [] # initial sample weight
            major_weight = 1
            for i in y_train:
                initial_weight = class_ratio_map[i]
                balanced_major_weight = minor * (class_ratio_map[i]==major_weight)
                balanced_minor_weight = major * (class_ratio_map[i]!=major_weight)
                new_weight =  initial_weight +  balanced_major_weight + balanced_minor_weight
                sample_weight.append(new_weight)
            sample_weight = np.array(sample_weight) # update the sample weight
            # Hyperparameter tuning the learning rate and batch size
            for lr in learning_rates:
                for bs in batch_sizes:
                    tf.keras.backend.clear_session() 

                    # Get the model
                    get_bayesian_classifier(model, lr)

                    # Train and load the best model
                    loaded_model = train_bayesian_classifier(model=model, 
                                            epochs=epochs,
                                            sample_weight=sample_weight, 
                                            batch_size=bs, 
                                            fpath_ckp=fpath_ckp,
                                            train_input=train_input,
                                            val_input=val_input,
                                            train_binary_labels=train_binary_labels,
                                            val_binary_labels=val_binary_labels)

                    # Evaluate the model on val set
                    results = loaded_model.evaluate(val_input, val_binary_labels, verbose=0)
                    auc = results[1] # get auc score

                    # Save the model that gives the best auc
                    if auc > best_auc:
                        print(f'Better AUC found is {round(auc, 4)}')
                        best_auc = auc
                        best_loaded_model = loaded_model
                        print('Next best model found!')
                    else:
                        print(f'AUC is {round(auc, 4)}')
                        print('Finished one training but havent found the best model. Tuning continued...')
                        continue

    print('Model training finished! Best AUC found. Now save the model weights.')
    # Save the best model
    best_loaded_model.save_weights(fpath_ckp) 
    # Load the best model
    best_loaded_model.load_weights(fpath_ckp) 

    return best_loaded_model

def tune_ood_model(
        train_input,
        val_input,
        y_train,
        train_binary_labels,
        val_binary_labels,
        model,
        epochs,
        learning_rates, 
        batch_sizes, 
        minor_weights, 
        major_weights, 
        fpath_ckp
    ):
    """Tune the sample weight and save the best model."""
    best_auc = 0 # initial auc

    # Hyperparameter tuning the sample weight
    for minor in minor_weights:
        for major in major_weights:            
            # Assign a new weight to each of the class in the train set
            sample_weight = [] # initial sample weight
            for i in y_train:
                if i==0:
                    sample_weight.append(15+minor)  
                elif i==3:
                    sample_weight.append(3+minor)
                elif i==2:
                    sample_weight.append(15+minor)
                else:
                    sample_weight.append(1+major)
            sample_weight = np.array(sample_weight) # update the sample weight

            # Hyperparameter tuning the learning rate and batch size
            for lr in learning_rates:
                for bs in batch_sizes:
                    tf.keras.backend.clear_session() 

                    # Get the model
                    get_bayesian_classifier(model, lr)

                    # Train and load the best model
                    loaded_model = train_bayesian_classifier(model=model, 
                                            epochs=epochs,
                                            sample_weight=sample_weight, 
                                            batch_size=bs, 
                                            fpath_ckp=fpath_ckp,
                                            train_input=train_input,
                                            val_input=val_input,
                                            train_binary_labels=train_binary_labels,
                                            val_binary_labels=val_binary_labels)

                    # Evaluate the model on val set
                    results = loaded_model.evaluate(val_input, val_binary_labels, verbose=0)
                    auc = results[1] # get auc score

                    # Save the model that gives the best auc
                    if auc > best_auc:
                        print(f'Better AUC found is {round(auc, 4)}')
                        best_auc = auc
                        best_loaded_model = loaded_model
                        print('Next best model found!')
                    else:
                        print(f'AUC is {round(auc, 4)}')
                        print('Finished one training but havent found the best model. Tuning continued...')
                        continue
    
    print('Model training finished! Best AUC found. Now save the model weights.')
    # Save the best model
    best_loaded_model.save_weights(fpath_ckp) 
    # Load the best model
    best_loaded_model.load_weights(fpath_ckp) 

    return best_loaded_model

def tune_id_model(
        train_input,
        val_input,
        y_train,
        train_binary_labels,
        val_binary_labels,
        model,
        epochs,
        learning_rate, 
        batch_size, 
        minor_weights, 
        major_weights, 
        major_class_ratios,
        fpath_ckp
    ):
    """Tune the sample weight and save the best model."""
    best_auc = 0 # initial auc

    # Hyperparameter tuning sample weight
    for minor in minor_weights:
        for major in major_weights:            
            # Assign a new weight to each of the class in the train set
            sample_weight = [] # initial sample weight
            for i in y_train:
                weight = major_class_ratios.loc[i].iloc[0,0]*major + (major_class_ratios.loc[i].iloc[0,0]-1)*minor
                sample_weight.append(weight)
            sample_weight = np.array(sample_weight) # update the sample weight

            # Start model training
            tf.keras.backend.clear_session() 

            # Get the model
            get_bayesian_classifier(model, learning_rate)

            # Train and load the best model
            loaded_model = train_bayesian_classifier(
                model=model, 
                epochs=epochs,
                sample_weight=sample_weight, 
                batch_size=batch_size, 
                fpath_ckp=fpath_ckp,
                train_input=train_input,
                val_input=val_input,
                train_binary_labels=train_binary_labels,
                val_binary_labels=val_binary_labels
            )

            # Evaluate the model on val set
            results = loaded_model.evaluate(val_input, val_binary_labels, verbose=0)
            auc = results[1] # get auc score

            # Save the model that gives the best auc
            if auc > best_auc:
                print(f'Better AUC found is {round(auc, 4)}')
                best_auc = auc
                best_loaded_model = loaded_model
                print('Next best model found!')
            else:
                print(f'AUC is {round(auc, 4)}')
                print('Finished one training but havent found the best model. Tuning continued...')
                continue

    print('Model training finished! Best AUC found. Now save the model weights.')
    # Save the best model
    best_loaded_model.save_weights(fpath_ckp) 
    # Load the best model
    best_loaded_model.load_weights(fpath_ckp) 
                   
    return best_loaded_model


if __name__=='__main__':
    train_model()
    train_bayesian_classifier()
    tune_ood_model()
    tune_id_model()
