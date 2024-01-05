import numpy as np
import pandas as pd
import keras
import random
import json
import os
from joblib import Parallel, delayed
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from utils import *


class Params:

    """
    Class for the simulation paramters
    """

    def __init__(self):
        """
        initialization of parameters with default value
        """
        self.n_epochs = 50
        self.batch_size = 32
        self.shape_data = 4  ## as 4 assets
        self.lr = 0.005
        self.b1 = 0.5
        self.b2 = 0.999
        self.latent_dim = 100
        self.sample_interval = 10


class Generator:
    """
    Class to define the Generator

    Main arguments are :

    - latent dimension : latent dimension of the raw noise, multiplied by 3 as we add cube and square of noise
    - number of layers (int)
    - number of neurons per layer (int)
    - hidden_activation between layers (string) exemple "leaky_relu"
    - activation function (string) ex "sigmoid"

    """

    def __init__(
        self,
        latent_dim,
        output_shape,
        num_layers,
        neurons_per_layer,
        hidden_activation,
        output_activation,
    ):
        self.latent_dim = latent_dim * 3
        self.output_shape = output_shape
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.model = self.build_model()

    def build_model(self):
        """
        Build the generator according to the parameters

        A batch normalization layer is added between each hidden layer

        """
        model = Sequential()
        model.add(
            Dense(
                self.neurons_per_layer[0],
                activation=self.hidden_activation[0],
                input_shape=(self.latent_dim,),
            )
        )
        for i in range(1, self.num_layers):
            model.add(BatchNormalization())
            model.add(
                Dense(self.neurons_per_layer[i], activation=self.hidden_activation[i])
            )
        model.add(BatchNormalization())
        model.add(Dense(self.output_shape, activation=self.output_activation))
        return model


class Discriminator:
    """
    Class to define the Generator

    Main arguments are :

    - input_shape : 4 as the Generator output 4 x1 arrays
    - number of layers (int)
    - number of neurons per layer (int)
    - hidden_activation between layers (string) exemple "leaky_relu"
    - activation function (string) ex "sigmoid"

    """

    def __init__(
        self,
        input_shape,
        num_layers,
        neurons_per_layer,
        hidden_activation,
        output_activation,
    ):
        self.output_shape = 1
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.model = self.build_model()

    def build_model(self):
        """
        Build the discriminator according to the parameters

        A batch normalization layer is added between each hidden layer

        """
        model = Sequential()
        model.add(
            Dense(
                self.neurons_per_layer[0],
                activation=self.hidden_activation[0],
                input_shape=(self.input_shape,),
            )
        )
        for i in range(1, self.num_layers):
            model.add(BatchNormalization())
            model.add(
                Dense(self.neurons_per_layer[i], activation=self.hidden_activation[i])
            )
        model.add(Dense(self.output_shape, activation=self.output_activation))
        model.add(BatchNormalization())
        opt = Adam(learning_rate=0.005, beta_1=0.5)
        model.compile(optimizer=opt, loss="binary_crossentropy")
        return model


class GAN:
    """
    class to aggregate the generator and the discriminator to build the gan

    Arguments are

    - generator : an instance of the Generator class
    - discriminator : an instance of the Discriminator class
    - params : parameters for the GAN, an instance of the Params class

    """

    def __init__(self, generator, discriminator, opt):
        self.generator = generator
        self.discriminator = discriminator
        self.opt = opt
        self.adversarial_loss = "binary_crossentropy"
        self.model = self.compile_models()

    def compile_models(self):
        """
        Stacking the generator and the discriminator together to build the GAN

        setting the optimizer, according to Params class instance
        """
        model = Sequential()
        model.add(self.generator.model)
        model.add(self.discriminator.model)

        optimizer = Adam(lr=self.opt.lr, beta_1=self.opt.b1, beta_2=self.opt.b2)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        return model

    def generate_noise(self, n_samples):
        """
        function to generate noise

        Argument :

        - n samples (int) number of samples to generate

        output

        - n_samples * (latent_dim) array of noise

        Note that we augment the noise, i.e if the latent_dim in the Params
        class is 100, we augment the latent dim to 300, as we add square of noise and cube of noise

        The raw noise is drawn from a multivariate gaussian matrix with a Toeplitz correlation matrix (see report)
        """

        covariance_matrix = 0.75 ** np.abs(
            np.subtract.outer(
                np.arange(self.opt.latent_dim), np.arange(self.opt.latent_dim)
            )
        )

        noise = np.random.multivariate_normal(
            mean=np.zeros(self.opt.latent_dim), cov=covariance_matrix, size=n_samples
        )
        squared_noise = noise**2
        cube_noise = noise**3
        noise = np.concatenate([noise, squared_noise, cube_noise], axis=1)
        return noise

    def generate_fake_normal(self, n_samples):
        """
        Function to create synthetic data

        Args:

        - n_samples : the number of samples of synthetic data to generate

        Output :
        - n_samples * 4 array

        The function generate noise according to generate_noise method aqnd use the generator of the GAN to convert this noise into synthetic data


        """
        noise = self.generate_noise(n_samples)
        X = self.generator.model.predict(noise)
        y = np.zeros((n_samples, 1))
        return X, y

    def train_gan(self, df_train):
        """
        function to train GAN
        """

        bat_per_epo = int(df_train.shape[0] / self.opt.batch_size)
        for epoch in range(self.opt.n_epochs):
            for batch in range(bat_per_epo):
                real_samples = df_train.values[
                    np.random.choice(
                        df_train.shape[0], self.opt.batch_size, replace=False
                    )
                ]
                fake_samples, labels_fake = self.generate_fake_normal(
                    self.opt.batch_size
                )

                labels_real = np.ones((self.opt.batch_size, 1))
                labels_fake = np.zeros((self.opt.batch_size, 1))

                real_loss = self.discriminator.model.train_on_batch(
                    real_samples, labels_real
                )
                fake_loss = self.discriminator.model.train_on_batch(
                    fake_samples, labels_fake
                )

                x_gan = self.generate_noise(self.opt.batch_size)
                y_gan = np.ones((self.opt.batch_size, 1))
                g_loss = self.model.train_on_batch(x_gan, y_gan)
            print(
                "[Epoch %d/%d] [D loss Fake: %f] [D loss real:%f] [G loss: %f]"
                % (epoch, self.opt.n_epochs, fake_loss, real_loss, g_loss)
            )

    def generate_syntethic_data(self, df_train):
        """
        redundance
        """
        synthetic_data, _ = self.generate_fake_normal(df_train.shape[0])
        synthetic_data = pd.DataFrame(synthetic_data, columns=df_train.columns)
        return synthetic_data

    def anderling_distance(self, data, predictions):
        """
        function to compute anderling distance between true distribution and predictions (synthetic data)
        """

        if isinstance(data, pd.DataFrame):
            data = data.values
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values

        N, P = data.shape
        ADdistance = 0
        for station in range(P):
            temp_predictions = predictions[:, station].reshape(-1)
            temp_data = data[:, station].reshape(-1)
            sorted_array = np.sort(temp_predictions)
            count = np.zeros(len(temp_data))
            count = (1 / (N + 2)) * np.array(
                [(temp_data < order).sum() + 1 for order in sorted_array]
            )
            idx = np.arange(1, N + 1)
            ADdistance = (2 * idx - 1) * (np.log(count) + np.log(1 - count[::-1]))
            ADdistance = -N - np.sum(ADdistance) / N
        return ADdistance / P


def generate_combination_name(g_config, d_config, latent_dim):
    """
     This function  will take an argument a generator config, a discriminator config, a param class config

     and will come up with a name for the simulation

    example :


    Generator_8_leaky_relu_8_leaky_relu_softplus_Discriminator_8_leaky_relu_8_leaky_relu_sigmoid_LatentDim_100


    For a generator with
     - 2 layers
     - 8 neurons per layer
     - leaky relu hidden activation
     - softplus activation function

    A discriminator :
    - 2 layers
    - 8 neurons per layer
    - leaky relu hidden activation
    - softplus activation function

    A latent dim of 100 (300 in total with square noise and cube noise)


    """

    def create_layer_description(
        num_layers, neurons_per_layer, hidden_activations, output_activation
    ):
        layer_desc = []
        for i in range(num_layers):
            neurons = neurons_per_layer[i]
            activation = hidden_activations[i]
            layer_desc.append(f"{neurons}_{activation}")
        layer_desc.append(output_activation)
        return "_".join(layer_desc)

    # Create descriptions for generator and discriminator
    g_desc = create_layer_description(
        g_config["num_layers"],
        g_config["neurons_per_layer"],
        g_config["hidden_activation"],
        g_config["output_activation"],
    )
    d_desc = create_layer_description(
        d_config["num_layers"],
        d_config["neurons_per_layer"],
        d_config["hidden_activation"],
        d_config["output_activation"],
    )
    return f"Generator_{g_desc}_Discriminator_{d_desc}_LatentDim_{latent_dim}"


def evaluate_model(g_config, d_config, latent_dim, df_train, results_dir):
    """
    Function that will test a model with a generator config, discriminator config and given latent dim

    The function

    - create generator and discriminator classes instance with given configurations
    - create a combinaison name (combi_name)
    - stack the generator and the discriminator into a GAN class instance
    - train the GAN
    - test the fitted GAN on 50 iterations
    - compute the mean AD distance and Kendall distance
    - save results into the folder results/combi_name, the configuration, the results and model weights
    -
    """

    results_dir = "results"
    opt = Params()
    opt.latent_dim = latent_dim
    generator = Generator(latent_dim, output_shape=opt.shape_data, **g_config)
    discriminator = Discriminator(input_shape=opt.shape_data, **d_config)
    gan = GAN(generator, discriminator, opt)

    gan.train_gan(df_train)

    ad_distances = []
    kendall_tau_distances = []
    for _ in range(50):
        synthetic_data = gan.generate_syntethic_data(df_train)
        ad_dist = gan.anderling_distance(df_train, synthetic_data)
        kendall_tau = kendall_tau_distance(synthetic_data, df_train)

        ad_distances.append(ad_dist)
        kendall_tau_distances.append(kendall_tau)

    mean_ad_distance = np.mean(ad_distances)
    mean_kendall_tau = np.mean(kendall_tau_distances)
    comb_name = generate_combination_name(g_config, d_config, latent_dim)
    # Append results to DataFrame
    new_row = pd.DataFrame(
        [
            {
                "name": comb_name,
                "g_number_layer": g_config["num_layers"],
                "g_number_neuron": g_config["neurons_per_layer"],
                "g_hidden_activation": g_config["hidden_activation"],
                "g_output_activation": g_config["output_activation"],
                "d_number_layer": d_config["num_layers"],
                "d_number_neuron": d_config["neurons_per_layer"],
                "d_hidden_activation": d_config["hidden_activation"],
                "d_output_activation": d_config["output_activation"],
                "latent_dim": latent_dim,
                "mean_anderling_distance": mean_ad_distance,
                "mean_kendall_tau": mean_kendall_tau,
            }
        ]
    )

    model_params = {
        "generator": g_config,
        "discriminator": d_config,
        "latent_dim": latent_dim,
    }
    comb_dir = os.path.join(results_dir, comb_name)
    os.makedirs(comb_dir, exist_ok=True)
    new_row.to_csv(os.path.join(comb_dir, "results.csv"), index=False)
    generator.model.save_weights(os.path.join(comb_dir, "generator_weights.h5"))
    discriminator.model.save_weights(os.path.join(comb_dir, "discriminator_weights.h5"))
    with open(os.path.join(comb_dir, "model_params.json"), "w") as f:
        json.dump(model_params, f)


def grid_search(grid_params, df_train, results_dir="results"):
    """
    testing all combinaison of generator, discriminator and latent dim
    Using Parallel from Joblib to speed up the process

    """
    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start parallel computation
    Parallel(n_jobs=3)(
        delayed(evaluate_model)(g_config, d_config, latent_dim, df_train, results_dir)
        for g_config in grid_params["generator"]
        for d_config in grid_params["discriminator"]
        for latent_dim in grid_params["latent_dim"]
    )


# Main execution block with grid search
if __name__ == "__main__":
    generator_combinations = [
        {
            "num_layers": num_layers,
            "neurons_per_layer": [neurons] * num_layers,
            "hidden_activation": [activation] * num_layers,
            "output_activation": output_activation,
        }
        for num_layers in [1, 2]
        for neurons in [8, 16, 32, 64]
        for activation in ["relu", "leaky_relu", "softplus"]
        for output_activation in ["softplus"]
    ]
    # Discriminator Combinations
    discriminator_combinations = [
        {
            "num_layers": num_layers,
            "neurons_per_layer": [neurons] * num_layers,
            "hidden_activation": [activation] * num_layers,
            "output_activation": output_activation,
        }
        for num_layers in [1, 2]
        for neurons in [8, 16, 32, 64]
        for activation in [
            "relu",
            "leaky_relu",
            "softplus",
        ]
        for output_activation in [
            "sigmoid",
            "softplus",
        ]
    ]
    # Latent Dimension Combinations
    latent_dims = [20, 50, 100]

    # Combine all these into a single grid_params variable for the grid search
    grid_params = {
        "generator": generator_combinations,
        "discriminator": discriminator_combinations,
        "latent_dim": latent_dims,
    }

    df_train = (
        open_data()
    )  # Make sure this function is defined to load your training data
    df_train = df_train * 100
    grid_search(grid_params, df_train)
