from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Set

from pyquaternion import Quaternion
import numpy as np
from numpy import linalg
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap


TOTAL_TIMESTEPS_LIMIT = 39
# dimensions taken from https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
EGO_VEHICLE_LENGTH = 4.084  # m
EGO_VEHICLE_WIDTH = 1.73  # m


class WaymoAgentType(IntEnum):
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4


@dataclass
class AgentRecord:
    x: float = -1
    y: float = -1
    yaw: float = -1
    length: float = -1
    width: float = -1
    velocity_x: float = -1
    velocity_y: float = -1
    speed: float = -1
    is_sdc: bool = False
    category: str = ''
    valid: bool = False
    attributes: Set[str] = field(default_factory=lambda: set())

    def category_to_type(self):
        if self.category == '':
            return WaymoAgentType.UNSET
        if self.category.startswith('human.pedestrian'):
            return WaymoAgentType.PEDESTRIAN
        if self.category == 'vehicle.bicycle' and 'cycle.with_rider' in self.attributes:
            return WaymoAgentType.CYCLIST
        if self.category.startswith('vehicle') and self.category != 'vehicle.bicycle':
            return WaymoAgentType.VEHICLE
        return WaymoAgentType.OTHER

    def get_core_tuple(self):
        return self.x, self.y, self.yaw, self.speed, float(self.valid), self.length, self.width, self.velocity_x, self.velocity_y


def get_scene_samples(nuscenes, scene):
    curr_sample = nuscenes.get('sample', scene['first_sample_token'])
    samples = [curr_sample]
    while curr_sample['next'] != '':
        curr_sample = nuscenes.get('sample', curr_sample['next'])
        samples.append(curr_sample)
    return samples


def get_agents_data(nuscenes, annotation_tokens):
    agent_id_to_data = {}
    for sample_annotation_token in annotation_tokens:
        sample_annotation = nuscenes.get('sample_annotation', sample_annotation_token)

        x, y = sample_annotation['translation'][:2]
        rotation_quaternion = Quaternion(sample_annotation['rotation'])
        width, length = sample_annotation['size'][:2]
        velocity_vector = nuscenes.box_velocity(sample_annotation_token)

        is_valid = not np.isnan(velocity_vector).any()
        if is_valid:
            velocity_x, velocity_y = velocity_vector[:2]
            speed = linalg.norm(velocity_vector)
        else:
            velocity_x, velocity_y = -1.0, -1.0
            speed = -1.0

        attributes = {nuscenes.get('attribute', attribute_token)['name'] for attribute_token in
                      sample_annotation['attribute_tokens']}

        agent_record = AgentRecord(
            x=x,
            y=y,
            yaw=quaternion_yaw(rotation_quaternion),
            length=length,
            width=width,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            is_sdc=False,
            speed=speed,
            category=sample_annotation['category_name'],
            valid=is_valid,
            attributes=attributes,
        )

        agent_id_to_data[sample_annotation['instance_token']] = agent_record

    return agent_id_to_data


def get_scene_samples_data(nuscenes, scene):
    scene_samples_data = []
    for sample in get_scene_samples(nuscenes, scene):
        sample_agents_data = get_agents_data(nuscenes, sample['anns'])
        sample_ego_vehicle_record = get_ego_vehicle_data(nuscenes, sample)
        sample_agents_data['ego-vehicle'] = sample_ego_vehicle_record
        scene_samples_data.append(sample_agents_data)

    return scene_samples_data


def get_ego_vehicle_data(nuscenes, sample):
    sample_data_token = sample['data']['LIDAR_TOP']
    sample_data = nuscenes.get('sample_data', sample_data_token)

    ego_pose = nuscenes.get('ego_pose', sample_data['ego_pose_token'])

    x, y = ego_pose['translation'][:2]

    velocity_vector = compute_ego_vehicle_velocity(nuscenes, sample)
    if velocity_vector is None:
        raise RuntimeError("cannot compute ego vehicle velocity: missing both prev and next sample")
    velocity_x, velocity_y = velocity_vector
    speed = linalg.norm(velocity_vector)

    ego_vehicle_record = AgentRecord(
        x=x,
        y=y,
        yaw=quaternion_yaw(Quaternion(ego_pose['rotation'])),
        length=EGO_VEHICLE_LENGTH,
        width=EGO_VEHICLE_WIDTH,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        is_sdc=True,
        speed=speed,
        category='vehicle.ego',
        valid=True,
    )

    return ego_vehicle_record


def compute_ego_vehicle_velocity(nuscenes, sample):
    has_prev = sample['prev'] != ''
    has_next = sample['next'] != ''

    if not has_prev and not has_next:
        return None
    
    if has_prev:
        first = nuscenes.get('sample', sample['prev'])
    else:
        first = sample
    if has_next:
        last = nuscenes.get('sample', sample['next'])
    else:
        last = sample

    first_sample_data = nuscenes.get('sample_data', first['data']['LIDAR_TOP'])
    last_sample_data = nuscenes.get('sample_data', last['data']['LIDAR_TOP'])

    first_ego_pos = np.array(nuscenes.get('ego_pose', first_sample_data['ego_pose_token'])['translation'][:2])
    last_ego_pos = np.array(nuscenes.get('ego_pose', last_sample_data['ego_pose_token'])['translation'][:2])

    pos_diff = last_ego_pos - first_ego_pos
    time_diff = 1e-6 * (last_sample_data['timestamp'] - first_sample_data['timestamp'])

    return pos_diff / time_diff


@dataclass
class SceneBoundingBox:
    x_min: float = -1
    y_min: float = -1
    x_max: float = -1
    y_max: float = -1

    def extend_by_radius(self, r):
        self.x_min -= r
        self.y_min -= r
        self.x_max += r
        self.y_max += r

    def as_patch(self):
        return self.x_min, self.y_min, self.x_max, self.y_max


def mock_data_continuously(data):
    """
    Converts [-1, 2, 2, -1, -1, 5, 5, -1] to [2, 2, 2, 2, 2, 5, 5, 5]
    """
    if (data == -1).all():
        return data

    idx = np.where(data != -1, np.arange(len(data)), 0)
    np.maximum.accumulate(idx, out=idx)
    data = data[idx]

    first = np.where(data != -1)[0][0]
    data[:first] = data[first]
    return data


def scene_data_to_agents_timesteps_dict(scene_id, scene_samples_data, current_timestep_idx):
    num_timesteps_total = min(TOTAL_TIMESTEPS_LIMIT, len(scene_samples_data))

    agent_to_timestep_to_data = defaultdict(lambda: [AgentRecord()] * num_timesteps_total)
    for timestep in range(num_timesteps_total):
        for agent_id, agent_record in scene_samples_data[timestep].items():
            agent_to_timestep_to_data[agent_id][timestep] = agent_record

    num_agents = len(agent_to_timestep_to_data)

    num_timesteps_history = current_timestep_idx
    num_timesteps_future = num_timesteps_total - num_timesteps_history - 1

    result = {
        'scenario/id': np.array(str(scene_id).encode('utf-8')),
        'state/id': np.empty(num_agents),
        'state/is_sdc': np.empty(num_agents),
        'state/type': np.empty(num_agents),
        'state/tracks_to_predict': np.empty(num_agents),

        'state/past/x': np.empty((num_agents, num_timesteps_history)),
        'state/past/y': np.empty((num_agents, num_timesteps_history)),
        'state/past/bbox_yaw': np.empty((num_agents, num_timesteps_history)),
        'state/past/speed': np.empty((num_agents, num_timesteps_history)),
        'state/past/valid': np.empty((num_agents, num_timesteps_history)),
        'state/past/length': np.empty((num_agents, num_timesteps_history)),
        'state/past/width': np.empty((num_agents, num_timesteps_history)),
        'state/past/velocity_x': np.empty((num_agents, num_timesteps_history)),
        'state/past/velocity_y': np.empty((num_agents, num_timesteps_history)),

        'state/current/x': np.empty((num_agents, 1)),
        'state/current/y': np.empty((num_agents, 1)),
        'state/current/bbox_yaw': np.empty((num_agents, 1)),
        'state/current/speed': np.empty((num_agents, 1)),
        'state/current/valid': np.empty((num_agents, 1)),
        'state/current/length': np.empty((num_agents, 1)),
        'state/current/width': np.empty((num_agents, 1)),
        'state/current/velocity_x': np.empty((num_agents, 1)),
        'state/current/velocity_y': np.empty((num_agents, 1)),

        'state/future/x': np.empty((num_agents, num_timesteps_future)),
        'state/future/y': np.empty((num_agents, num_timesteps_future)),
        'state/future/bbox_yaw': np.empty((num_agents, num_timesteps_future)),
        'state/future/speed': np.empty((num_agents, num_timesteps_future)),
        'state/future/valid': np.empty((num_agents, num_timesteps_future)),
        'state/future/length': np.empty((num_agents, num_timesteps_future)),
        'state/future/width': np.empty((num_agents, num_timesteps_future)),
        'state/future/velocity_x': np.empty((num_agents, num_timesteps_future)),
        'state/future/velocity_y': np.empty((num_agents, num_timesteps_future)),
    }

    for agent_idx, (agent_id, agent_records) in enumerate(agent_to_timestep_to_data.items()):
        result['state/id'][agent_idx] = agent_idx

        result['state/is_sdc'][agent_idx] = any(record.is_sdc for record in agent_records)

        agent_type = {record.category_to_type()
                      for record in agent_records if record.category_to_type() != WaymoAgentType.UNSET}
        assert len(agent_type) <= 1
        result['state/type'][agent_idx] = WaymoAgentType.UNSET if len(agent_type) == 0 else agent_type.pop()

        result['state/tracks_to_predict'][agent_idx] = agent_idx if agent_records[current_timestep_idx].valid else 0.0

        agent_records_core = np.array([agent_record.get_core_tuple() for agent_record in agent_records])

        # mock all the data expect validity
        # we actually only need to mock x and y because it affects road embeddings retrieval
        for i in range(9):
            if i == 4:
                continue
            agent_records_core[:, i] = mock_data_continuously(agent_records_core[:, i])

        result['state/past/x'][agent_idx] = agent_records_core[:current_timestep_idx, 0]
        result['state/past/y'][agent_idx] = agent_records_core[:current_timestep_idx, 1]
        result['state/past/bbox_yaw'][agent_idx] = agent_records_core[:current_timestep_idx, 2]
        result['state/past/speed'][agent_idx] = agent_records_core[:current_timestep_idx, 3]
        result['state/past/valid'][agent_idx] = agent_records_core[:current_timestep_idx, 4]
        result['state/past/length'][agent_idx] = agent_records_core[:current_timestep_idx, 5]
        result['state/past/width'][agent_idx] = agent_records_core[:current_timestep_idx, 6]
        result['state/past/velocity_x'][agent_idx] = agent_records_core[:current_timestep_idx, 7]
        result['state/past/velocity_y'][agent_idx] = agent_records_core[:current_timestep_idx, 8]

        result['state/current/x'][agent_idx] = agent_records_core[current_timestep_idx, 0]
        result['state/current/y'][agent_idx] = agent_records_core[current_timestep_idx, 1]
        result['state/current/bbox_yaw'][agent_idx] = agent_records_core[current_timestep_idx, 2]
        result['state/current/speed'][agent_idx] = agent_records_core[current_timestep_idx, 3]
        result['state/current/valid'][agent_idx] = agent_records_core[current_timestep_idx, 4]
        result['state/current/length'][agent_idx] = agent_records_core[current_timestep_idx, 5]
        result['state/current/width'][agent_idx] = agent_records_core[current_timestep_idx, 6]
        result['state/current/velocity_x'][agent_idx] = agent_records_core[current_timestep_idx, 7]
        result['state/current/velocity_y'][agent_idx] = agent_records_core[current_timestep_idx, 8]

        result['state/future/x'][agent_idx] = agent_records_core[current_timestep_idx+1:, 0]
        result['state/future/y'][agent_idx] = agent_records_core[current_timestep_idx+1:, 1]
        result['state/future/bbox_yaw'][agent_idx] = agent_records_core[current_timestep_idx+1:, 2]
        result['state/future/speed'][agent_idx] = agent_records_core[current_timestep_idx+1:, 3]
        result['state/future/valid'][agent_idx] = agent_records_core[current_timestep_idx+1:, 4]
        result['state/future/length'][agent_idx] = agent_records_core[current_timestep_idx+1:, 5]
        result['state/future/width'][agent_idx] = agent_records_core[current_timestep_idx+1:, 6]
        result['state/future/velocity_x'][agent_idx] = agent_records_core[current_timestep_idx+1:, 7]
        result['state/future/velocity_y'][agent_idx] = agent_records_core[current_timestep_idx+1:, 8]

    current_valid_mask = result['state/current/valid'] > 0
    x_min = result['state/current/x'].min(initial=0, where=current_valid_mask)
    y_min = result['state/current/y'].min(initial=0, where=current_valid_mask)
    x_max = result['state/current/x'].max(initial=0, where=current_valid_mask)
    y_max = result['state/current/y'].max(initial=0, where=current_valid_mask)

    scene_bounding_box = SceneBoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )

    return result, scene_bounding_box


class RoadgraphLayerType(IntEnum):
    ROAD_SEGMENT = 1
    ROAD_BLOCK = 2
    LANE = 3
    PED_CROSSING = 4
    WALKWAY = 5
    STOP_LINE = 6
    CARPARK_AREA = 7
    ROAD_DIVIDER = 8
    LANE_DIVIDER = 9


def roadgraph_layer_string_to_enum(layer):
    if layer == 'road_segment':
        return RoadgraphLayerType.ROAD_SEGMENT
    if layer == 'road_block':
        return RoadgraphLayerType.ROAD_BLOCK
    if layer == 'lane':
        return RoadgraphLayerType.LANE
    if layer == 'ped_crossing':
        return RoadgraphLayerType.PED_CROSSING
    if layer == 'walkway':
        return RoadgraphLayerType.WALKWAY
    if layer == 'stop_line':
        return RoadgraphLayerType.STOP_LINE
    if layer == 'carpark_area':
        return RoadgraphLayerType.CARPARK_AREA
    if layer == 'road_divider':
        return RoadgraphLayerType.ROAD_DIVIDER
    if layer == 'lane_divider':
        return RoadgraphLayerType.LANE_DIVIDER
    raise RuntimeError(f'Unknown layer: {layer}')


def get_scene_map(nuscenes, scene):
    scene_location = nuscenes.get('log', scene['log_token'])['location']
    return NuScenesMap(dataroot=nuscenes.dataroot, map_name=scene_location)


def get_scene_roadgraph(nusc_map, scene_bbox, r, layers_of_interest):
    scene_bbox.extend_by_radius(r)

    node_coordinates = []
    node_object_ids = []
    node_types = []

    for layer_name, layer_object_tokens in nusc_map.get_records_in_patch(scene_bbox.as_patch(), layer_names=layers_of_interest).items():
        for object_token in layer_object_tokens:
            layer_objects = nusc_map.get(layer_name, object_token)

            node_tokens = layer_objects.get('exterior_node_tokens', []) + layer_objects.get('node_tokens', [])
            for node_token in node_tokens:
                node_data = nusc_map.get('node', node_token)
                x, y = node_data['x'], node_data['y']

                node_coordinates.append((x, y))
                node_object_ids.append(object_token)
                node_types.append(int(roadgraph_layer_string_to_enum(layer_name)))

    return {
        'roadgraph_samples/xyz': np.array(node_coordinates),
        'roadgraph_samples/id': np.array(node_object_ids),
        'roadgraph_samples/type': np.array(node_types),
        'roadgraph_samples/valid': np.ones(len(node_coordinates)),
    }


def get_full_scene_data(nuscenes, config, scene_id):
    scene = nuscenes.scene[scene_id]

    scene_samples_data = get_scene_samples_data(nuscenes, scene)
    agents_dict, scene_bbox = scene_data_to_agents_timesteps_dict(scene_id, scene_samples_data, config["current_timestep_idx"])

    scene_map = get_scene_map(nuscenes, scene)
    roadgraph_dict = get_scene_roadgraph(scene_map, scene_bbox, config["map_expansion_radius"],
                                         config["layers_of_interest"])

    agents_dict.update(roadgraph_dict)

    return agents_dict
