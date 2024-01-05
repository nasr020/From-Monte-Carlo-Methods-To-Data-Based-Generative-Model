import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def aggregate_results(results_dir="results"):
    all_results = []
    # Loop through each combination folder in the results directory
    for combi_name in os.listdir(results_dir):
        combi_dir = os.path.join(results_dir, combi_name)
        if os.path.isdir(combi_dir):
            csv_file = os.path.join(combi_dir, "results.csv")
            if os.path.exists(csv_file):
                # Read the CSV file and append it to the list
                df = pd.read_csv(csv_file)
                df[
                    "Combination"
                ] = combi_name  # Optionally, add a column indicating the combination
                all_results.append(df)

    # Concatenate all dataframes into one
    df = pd.concat(all_results, ignore_index=True)
    df = df.sort_values(by="mean_anderling_distance")
    return df


def open_data():
    file_path = "data_train_log_return.csv"
    header = ["stock1", "stock2", "stock3", "stock4"]
    df_train = pd.read_csv(file_path, header=None, index_col=0)
    df_train.columns = header

    return df_train


def kendall_tau_distance(df1, df2):
    tau_dist = 0
    for i in range(len(df1.columns)):
        for j in range(i + 1, len(df1.columns)):
            tau_dist += np.abs(
                np.mean(
                    np.sign(df1[df1.columns[i]] - df1[df1.columns[j]])
                    * np.sign(df2[df2.columns[i]] - df2[df2.columns[j]])
                )
            )
    return tau_dist / (len(df1.columns) * (len(df1.columns) - 1) / 2)


def AndersonDarling(data, predictions):
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


def cholesky_synthetic_data(num_samples, correlation_matrix):
    cholesky_matrix = np.linalg.cholesky(correlation_matrix)
    synthetic_data = (
        np.random.normal(0, 1, size=(num_samples, len(correlation_matrix)))
        @ cholesky_matrix.T
    )
    return synthetic_data


def filter_positive_extremes(synthetic_data, significance_level=0.95):
    theoretical_percentile = np.percentile(
        np.random.normal(0, 1, size=(100000, synthetic_data.shape[1])),
        significance_level * 100,
        axis=0,
    )

    positive_filter = np.all(
        (synthetic_data < theoretical_percentile) & (synthetic_data > 0), axis=1
    )
    filtered_data = synthetic_data[positive_filter]

    return filtered_data


def adjust_for_correlation(synthetic_data, original_data):
    target_correlation_matrix = 2 * original_data.corr()
    target_cholesky = np.linalg.cholesky(target_correlation_matrix)
    synthetic_correlation_matrix = np.corrcoef(synthetic_data, rowvar=False)

    synthetic_cholesky = np.linalg.cholesky(synthetic_correlation_matrix)
    adjusted_synthetic_data = (
        synthetic_cholesky @ np.linalg.inv(target_cholesky) @ synthetic_data.T
    )
    adjusted_synthetic_data = adjusted_synthetic_data.T
    return adjusted_synthetic_data[
        np.all(adjusted_synthetic_data > 0, axis=1)
    ]  ## justin case


def cholesky(target_sample_size, cov_matrix):
    current_sample_size = 0
    while current_sample_size < target_sample_size:
        synthetic_batch = cholesky_synthetic_data(target_sample_size, cov_matrix)
        synthetic_batch = filter_positive_extremes(
            synthetic_batch, significance_level=0.99
        )
        current_batch_size = synthetic_batch.shape[0]
        if current_sample_size + current_batch_size <= target_sample_size:
            if current_sample_size == 0:
                synthetic_data = synthetic_batch
            else:
                synthetic_data = np.vstack([synthetic_data, synthetic_batch])
        else:
            synthetic_data = np.vstack(
                [
                    synthetic_data,
                    synthetic_batch[: target_sample_size - current_sample_size],
                ]
            )
        current_sample_size = synthetic_data.shape[0]
    return synthetic_data


def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.normal(0, 1, latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def compute_cdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y


def plot_cdf(df_train, synthetic_data):
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Flatten the axis array for easy iteration
    axs = axs.ravel()

    for i, column in enumerate(df_train.columns):
        # Compute CDFs
        x_train, y_train = compute_cdf(df_train[column])
        x_synthetic, y_synthetic = compute_cdf(synthetic_data[column])

        # Plot CDFs
        axs[i].plot(x_train, y_train, label="True Distribution", color="blue")
        axs[i].plot(
            x_synthetic, y_synthetic, label="Synthetic Distribution", color="red"
        )

        axs[i].set_title(f"CDF of {column}")
        axs[i].set_xlabel("Value")
        axs[i].set_ylabel("CDF")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def compare(synthetic_data, df_train):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Iterate through columns and plot for each subplot
    for i, column_name in enumerate(df_train.columns):
        row_index = i // 2
        col_index = i % 2
        sns.histplot(
            df_train[column_name],
            kde=True,
            label="Original Data",
            stat="density",
            color="blue",
            alpha=0.5,
            ax=axes[row_index, col_index],
        )
        sns.histplot(
            synthetic_data[column_name],
            kde=True,
            label="Synthetic Data",
            stat="density",
            color="orange",
            alpha=0.5,
            ax=axes[row_index, col_index],
        )
        axes[row_index, col_index].set_title(f"{column_name} Distribution")
        axes[row_index, col_index].set_xlabel(column_name)
        axes[row_index, col_index].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
    sns.heatmap(
        df_train.corr(),
        fmt=".2f",
        annot=True,
        cmap=sns.diverging_palette(h_neg=20, h_pos=220),
        center=0,
    ).set(title="Original data - Log returns correlation")

    plt.tight_layout()
    plt.show()

    sns.heatmap(
        synthetic_data.corr(),
        fmt=".2f",
        annot=True,
        cmap=sns.diverging_palette(h_neg=20, h_pos=220),
        center=0,
    ).set(title="synthetic data - Log returns correlation")
