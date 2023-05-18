import argparse
import numpy as np

from tqdm import tqdm

from model.data import MultiPathPPDataset
from prerender.utils.utils import get_config


def calculate_normalization_coefficients(
        dataset,
        history_timesteps,
        future_timesteps,
        agent_feature_count,
        agent_diff_feature_count,
        road_network_feature_count,
        feature_dimension_map,
        respect_validity
):

    total_ds = {
        'target/history/agent_features': np.empty((0, agent_feature_count)),
        'other/history/agent_features': np.empty((0, agent_feature_count)),
        'target/history/agent_features_diff': np.empty((0, agent_diff_feature_count)),
        'other/history/agent_features_diff': np.empty((0, agent_diff_feature_count)),
        'road_network_embeddings': np.empty((0, road_network_feature_count)),
        'target/future/xy': np.empty((0, 2))
    }

    for i in tqdm(range(len(dataset))):
        value = dataset.get_item_with_retries(i)
        if value is None:
            continue

        target_history_lstm_value = value['target/history/lstm_data'][:, :, :agent_feature_count]
        total_ds['target/history/agent_features'] = np.vstack((
            total_ds['target/history/agent_features'],
            (
                target_history_lstm_value[value['target/history/valid'].squeeze(axis=2) > 0] if respect_validity else
                target_history_lstm_value
            )
        ))

        other_history_lstm_value = value['other/history/lstm_data'][:, :, :agent_feature_count]
        total_ds['other/history/agent_features'] = np.vstack((
            total_ds['other/history/agent_features'],
            (
                other_history_lstm_value[value['other/history/valid'].squeeze(axis=2) > 0] if respect_validity else
                other_history_lstm_value
            )
        ))

        target_history_lstm_diff_value = value['target/history/lstm_data_diff'][:, :, :agent_diff_feature_count]
        total_ds['target/history/agent_features_diff'] = np.vstack((
            total_ds['target/history/agent_features_diff'],
            (
                target_history_lstm_diff_value[value['target/history/valid_diff'].squeeze(axis=2) > 0] if respect_validity else
                target_history_lstm_diff_value
            )
        ))

        other_history_lstm_diff_value = value['other/history/lstm_data_diff'][:, :, :agent_diff_feature_count]
        total_ds['other/history/agent_features_diff'] = np.vstack((
            total_ds['other/history/agent_features_diff'],
            (
                other_history_lstm_diff_value[value['other/history/valid_diff'].squeeze(axis=2) > 0] if respect_validity else
                other_history_lstm_diff_value
            )
        ))

        total_ds['road_network_embeddings'] = np.vstack((
            total_ds['road_network_embeddings'],
            value['road_network_embeddings'][:, :, :road_network_feature_count].squeeze()
        ))

        target_future_xy_value = value['target/future/xy']
        total_ds['target/future/xy'] = np.vstack((
            total_ds['target/future/xy'],
            (
                target_future_xy_value[value['target/future/valid'].squeeze(axis=2) > 0] if respect_validity else
                target_future_xy_value
            )
        ))

    means = {}
    stds = {}

    for k, v in total_ds.items():
        means[k] = np.mean(total_ds[k], axis=0)
        stds[k] = np.std(total_ds[k], axis=0)

    def _feature_key_to_aggregation_key(key):
        if key == 'target/history/lstm_data':
            return 'target/history/agent_features'
        if key == 'target/history/mcg_input_data':
            return 'target/history/agent_features'

        if key == 'other/history/lstm_data':
            return 'other/history/agent_features'
        if key == 'other/history/mcg_input_data':
            return 'other/history/agent_features'

        if key == 'target/history/lstm_data_diff':
            return 'target/history/agent_features_diff'
        if key == 'other/history/lstm_data_diff':
            return 'other/history/agent_features_diff'

        return key

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
    config['dataset_config']['data_path'] = args.data_path

    dataset = MultiPathPPDataset(config["dataset_config"])

    result = calculate_normalization_coefficients(
        dataset,
        history_timesteps=config['history_timesteps'],
        future_timesteps=config['future_timesteps'],
        agent_feature_count=config['agent_feature_count'],
        agent_diff_feature_count=config['agent_diff_feature_count'],
        road_network_feature_count=config['road_network_feature_count'],
        feature_dimension_map=config['feature_dimension_map'],
        respect_validity=config['respect_validity']
    )

    np.save(args.output_path, result)

    # Save result as YAML
    with open(args.output_path, 'w') as outfile:
        yaml.dump(result, outfile, default_flow_style=False)


if __name__ == "__main__":
    main()
