import argparse
import numpy as np

from tqdm import tqdm

from code.model.data import MultiPathPPDataset
from code.prerender.utils.utils import get_config


def calculate_normalization_coefficients(
        dataset,
        timesteps,
        agent_feature_count,
        agent_diff_feature_count,
        road_network_feature_count,
        feature_dimension_map,
):

    total_ds = {
        'target/agent_features': np.empty((0, timesteps, agent_feature_count)),
        'other/agent_features': np.empty((0, timesteps, agent_feature_count)),
        'target/agent_features_diff': np.empty((0, timesteps - 1, agent_diff_feature_count)),
        'other/agent_features_diff': np.empty((0, timesteps - 1, agent_diff_feature_count)),
        'road_network_embeddings': np.empty((0, 1, road_network_feature_count))
    }

    for i in tqdm(range(len(dataset))):
        value = dataset.get_item_with_retries(i)
        if value is None:
            continue

        total_ds['target/agent_features'] = np.vstack((
            total_ds['target/agent_features'],
            value['target/history/lstm_data'][:, :, :agent_feature_count]
        ))
        total_ds['other/agent_features'] = np.vstack((
            total_ds['other/agent_features'],
            value['other/history/lstm_data'][:, :, :agent_feature_count]
        ))

        total_ds['target/agent_features_diff'] = np.vstack((
            total_ds['target/agent_features_diff'],
            value['target/history/lstm_data_diff'][:, :, :agent_diff_feature_count]
        ))
        total_ds['other/agent_features_diff'] = np.vstack((
            total_ds['other/agent_features_diff'],
            value['other/history/lstm_data_diff'][:, :, :agent_diff_feature_count]
        ))

        total_ds['road_network_embeddings'] = np.vstack((
            total_ds['road_network_embeddings'],
            value['road_network_embeddings'][:, :, :road_network_feature_count]
        ))

    means = {}
    stds = {}

    for k, v in total_ds.items():
        means[k] = np.mean(total_ds[k], axis=(0, 1))
        stds[k] = np.std(total_ds[k], axis=(0, 1))

    def _feature_key_to_aggregation_key(key):
        if key == 'target/history/lstm_data':
            return 'target/agent_features'
        if key == 'target/history/mcg_input_data':
            return 'target/agent_features'

        if key == 'other/history/lstm_data':
            return 'other/agent_features'
        if key == 'other/history/mcg_input_data':
            return 'other/agent_features'

        if key == 'target/history/lstm_data_diff':
            return 'target/agent_features_diff'
        if key == 'other/history/lstm_data_diff':
            return 'other/agent_features_diff'

        return 'road_network_embeddings'

    result = {'mean': {}, 'std': {}}
    for feature, dim in feature_dimension_map.items():
        mean = means[_feature_key_to_aggregation_key(feature)]
        std = stds[_feature_key_to_aggregation_key(feature)]

        result['mean'][feature] = np.concatenate([mean, np.zeros(dim - mean.size)])
        result['std'][feature] = np.concatenate([std, np.ones(dim - mean.size)])

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to pre-rendered data")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save normalizations")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    config = get_config(args.config)
    config['data_path'] = args.data_path

    dataset = MultiPathPPDataset(config["dataset_config"])

    result = calculate_normalization_coefficients(
        dataset,
        timesteps=config['history_timesteps'],
        agent_feature_count=config['agent_feature_count'],
        agent_diff_feature_count=config['agent_diff_feature_count'],
        road_network_feature_count=config['road_network_feature_count'],
        feature_dimension_map=config['feature_dimension_map'],
    )

    np.save(args.output_path, result)


if __name__ == "__main__":
    main()
