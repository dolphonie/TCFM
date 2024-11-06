import pandas as pd
import numpy as np
from typing import List, Tuple

def generate_normalization_params(
    df: pd.DataFrame,
    wanted_features_mean_std: List[Tuple[str, str]],
    wanted_features_min_max: List[Tuple[str, str]],
    output_file: str = 'normalization_params.txt',
    std_devs: float = 3.0
) -> None:
    """
    Generate normalization parameters for given features and save to a text file.

    Args:
    df (pd.DataFrame): Input DataFrame containing the features.
    wanted_features_mean_std (List[Tuple[str, str]]): List of tuples (feature_name, feature_description) for mean-std normalization.
    wanted_features_min_max (List[Tuple[str, str]]): List of tuples (feature_name, feature_description) for min-max normalization.
    output_file (str): Name of the output text file.
    std_devs (float): Number of standard deviations to use for mean-std normalization.
    """
    normalization_params = {}

    def process_feature(feature, use_mean_std):
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in the DataFrame. Skipping.")
            return None

        feature_data = df[feature].dropna()

        if feature_data.empty:
            print(f"Warning: No valid data for feature '{feature}'. Skipping.")
            return None

        min_val = feature_data.min()
        max_val = feature_data.max()

        if int(min_val) == int(max_val):
            print(f"Warning: Feature '{feature}' has constant value {min_val}. Skipping.")
            return None

        if use_mean_std:
            mean_val = feature_data.mean()
            std_val = feature_data.std()
            lower_bound = mean_val - std_devs * std_val
            upper_bound = mean_val + std_devs * std_val
        else:
            lower_bound = min_val
            upper_bound = max_val

        return {
            'min': float(lower_bound),
            'max': float(upper_bound)
        }

    for feature, _ in wanted_features_mean_std:
        params = process_feature(feature, use_mean_std=True)
        if params:
            normalization_params[feature] = params

    for feature, _ in wanted_features_min_max:
        params = process_feature(feature, use_mean_std=False)
        if params:
            normalization_params[feature] = params

    # Get all feature names
    all_features = list(normalization_params.keys())

    # Format the output string
    output_str = f"'packed_features': {all_features},\n\n"
    output_str += "'normalization': {\n"
    for feature, params in normalization_params.items():
        output_str += f"    '{feature}': {{'min': {params['min']:.2f}, 'max': {params['max']:.2f}}},\n"
    output_str += "},"

    # Write to file
    with open(output_file, 'w') as f:
        f.write(output_str)

    print(f"Normalization parameters and packed features saved to {output_file}")


if __name__ == "__main__":
    # Example usage:
    csv_file = 'data/march_flight_coamps.csv'
    df = pd.read_csv(csv_file)

    # Convert timestamp to Unix timestamp (assuming you have this function defined)
    # df['timestamp'] = df['timestamp'].apply(convert_timestamp)

    wanted_features_mean_std = [
        ("conpac_sfc", "Accumulated Convective Precipitation"),
        ("stapac_sfc", "Accumulated Stable Precipitation"),
        ("ttlpcp_sfc", "Accumulated Total Precipitation"),
        ("airtmp_sig", "Air Temperature"),
        ("cldmix_sig", "Cloud Mixing Ratio [sigm]"),
        ("cld_mix_zht_000000", "Cloud Mixing Ratio [ht_sfc]"),
        ("grpmix_sig", "Graupel Mixing Ratio"),
        ("icemix_sig", "Ice Mixing Ratio [sigm]"),
        ("icemix_zht_000000", "Ice Mixing Ratio [ht_sfc]"),
        ("cmpice_sig", "Icing potential [sigm]"),
        ("cmpice_zht_000000", "Icing potential [ht_sfc]"),
        ("lndsea_sfc", "Land Sea Table"),
        ("ranmix_sig", "Rain Mixing Ratio [sigm]"),
        ("ranmix_zht_0000000", "Rain Mixing Ratio [ht_sfc]"),
        ("relhum_sig", "Relative Humidity"),
        ("snomix_sig", "Snow Mixing Ratio [sigm]"),
        ("snomix_zht_0000000", "Snow Mixing Ratio [ht_sfc]"),
        ("terrht_sfc", "Terrain Height"),
        ("trpres_sfc", "Terrain Pressure"),
        ("ttlprs_sig", "Total Pressure"),
        ("uutrue_sig", "True U-Velocity Component [sigm]"),
        ("uutrue_zht_000000", "True U-Velocity Component [ht_sfc]"),
        ("vvtrue_sig", "True V-Velocity Component [sigm]"),
        ("vvtrue_zht_000000", "True V-Velocity Component [ht_sfc]"),
        ("turbke_sig", "Turbulent Kinetic Energy [sigm]"),
        ("turbke_zht_000000", "Turbulent Kinetic Energy [ht_sfc]"),
        ("wvapor_sig", "Water Vapor Mixing Ratio"),
    ]

    wanted_features_min_max = [
        ("longitude", "Longitude"),
        ("latitude", "Latitude"),
        ("altitude", "Altitude")
    ]

    generate_normalization_params(df, wanted_features_mean_std, wanted_features_min_max, std_devs=1.0)