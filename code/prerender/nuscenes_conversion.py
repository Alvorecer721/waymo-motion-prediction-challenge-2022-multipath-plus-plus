from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Set

from pyquaternion import Quaternion
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw


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
    width: float = -1
    length: float = -1
    category: str = ''
    valid: bool = False
    attributes: Set[str] = field(default_factory=lambda: set())

    def category_mapping(self):
        if self.category == '':
            return WaymoAgentType.UNSET
        if self.category.startswith('human.pedestrian'):
            return WaymoAgentType.PEDESTRIAN
        if self.category == 'vehicle.bicycle' and 'cycle.with_rider' in self.attributes:
            return WaymoAgentType.CYCLIST
        if self.category.startswith('vehicle') and self.category != 'vehicle.bicycle':
            return WaymoAgentType.VEHICLE
        return WaymoAgentType.OTHER

    def get_as_tuple(self):
        return self.x, self.y, self.yaw, self.width, self.length, int(self.category_mapping()), self.valid


def get_annotation_tokens_by_sample(nuscenes, scene):
    curr_sample = nuscenes.get('sample', scene['first_sample_token'])
    samples_with_annotations = [curr_sample['anns']]
    while curr_sample['next'] != '':
        curr_sample = nuscenes.get('sample', curr_sample['next'])
        samples_with_annotations.append(curr_sample['anns'])

    return samples_with_annotations


def get_agents_data(nuscenes, annotation_tokens):
    agent_id_to_data = {}
    for sample_annotation_token in annotation_tokens:
        sample_annotation = nuscenes.get('sample_annotation', sample_annotation_token)

        x, y = sample_annotation['translation'][:2]
        rotation_quaternion = Quaternion(sample_annotation['rotation'])
        width, length = sample_annotation['size'][:2]

        attributes = {nuscenes.get('attribute', attribute_token)['name'] for attribute_token in
                      sample_annotation['attribute_tokens']}

        agent_record = AgentRecord(
            x=x,
            y=y,
            yaw=quaternion_yaw(rotation_quaternion),
            width=width,
            length=length,
            category=sample_annotation['category_name'],
            valid=True,
            attributes=attributes,
        )

        agent_id_to_data[sample_annotation['instance_token']] = agent_record

    return agent_id_to_data


def get_scene_samples_data(nuscenes, scene):
    scene_samples_data = []
    for sample_annotation_tokens in get_annotation_tokens_by_sample(nuscenes, scene):
        sample_agents_data = get_agents_data(nuscenes, sample_annotation_tokens)
        scene_samples_data.append(sample_agents_data)
    return scene_samples_data


def get_scenes_data(nuscenes):
    scenes_data = []
    for scene in nuscenes.scene:
        scene_samples_data = get_scene_samples_data(nuscenes, scene)
        scenes_data.append(scene_samples_data)
    return scenes_data


def scene_data_to_agents_timesteps_array(scene_samples_data):
    timesteps_total = len(scene_samples_data)
    agent_to_timestep_to_data = defaultdict(lambda: [AgentRecord()] * timesteps_total)

    for timestep, agents_data in enumerate(scene_samples_data):
        for agent_id, agent_record in agents_data.items():
            agent_to_timestep_to_data[agent_id][timestep] = agent_record

    agents_total = len(agent_to_timestep_to_data)

    result = np.empty((agents_total, timesteps_total, 7))

    for agent_idx, (agent_id, agent_records) in enumerate(agent_to_timestep_to_data.items()):
        agent_records_tupled = [agent_record.get_as_tuple() for agent_record in agent_records]
        result[agent_idx] = np.array(agent_records_tupled)

    return result
