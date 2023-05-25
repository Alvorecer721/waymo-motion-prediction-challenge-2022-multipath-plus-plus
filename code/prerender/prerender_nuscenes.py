from tqdm import tqdm
from utils.prerender_utils import get_visualizers, merge_and_save
from utils.nuscenes_conversion import get_scene_samples_data, scene_data_to_agents_timesteps_dict, get_scene_map, get_scene_roadgraph
from utils.utils import get_config
from nuscenes import NuScenes
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", type=str, required=True, help="Path to nuscenes data")
    parser.add_argument("--data-path", type=str, required=True, help="Path to nuscenes data")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save data")
    parser.add_argument("--config", type=str, required=True, help="Config file path")

    nuscenes = NuScenes(args.data_version, dataroot=args.data_path)

    config = get_config(args.config)
    visualizers = get_visualizers(config["renderers"])

    for scene_id, scene in enumerate(tqdm(nuscenes.scene)):
        scene_samples_data = get_scene_samples_data(nuscenes, scene)
        agents_dict, scene_bbox = scene_data_to_agents_timesteps_dict(scene_id, scene_samples_data, config["current_timestep_idx"])

        scene_map = get_scene_map(nuscenes, scene)
        roadgraph_dict = get_scene_roadgraph(scene_map, scene_bbox, config["map_expansion_radius"], config["layers_of_interest"])

        agents_dict.update(roadgraph_dict)
        merge_and_save(visualizers=visualizers, data=agents_dict, output_path=args.output_path, is_nuscenes=True)


if __name__ == "__main__":
    main()
