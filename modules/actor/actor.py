import os
import sys
import json

import numpy as np
from actor import tasks
from actor import agents
from actor.utils import utils

import torch
import cv2
from actor.dataset import RavensDataset
from environments.environment import Environment
from actor.tasks import cameras
from actor.tasks import primitives
from actor.tasks.grippers import Suction
from actor.tasks.task import Task

import pybullet as p

oracle_cams = cameras.Oracle.CONFIG
# Workspace bounds.
pix_size = 0.003125
bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])

class GeneralTask(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "put the {pick} on {place}"
        self.task_completed_desc = "done."

    def reset(self, env):
        super().reset(env)

    def set_goals(self, env, line, obj_info):
    # objs, matches, targs, _, _, metric, params, max_reward = goals
    # object_id, (symmetry, _) = objs[i]
        idx = line.find('\',\'')
        obj = line[7: idx]
        targ = line[idx+3: -2]
        # print(obj, targ)
        obj_id = obj_info[obj]
        # Associate placement locations for goals.
        targ_pos = p.getBasePositionAndOrientation(obj_info[targ])
        self.goals = [([(obj_id, (np.pi / 2, None))],
                           np.ones((1, 1)),
                           [(utils.apply(targ_pos, (0, 0, 0.05)), targ_pos[1])],
                           False, True, 'pose', None, 1)]
        self.lang_goals =[self.lang_template.format(pick=obj,
                                                    place=targ)]

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


def get_true_image(env):
    """Get RGB-D orthographic heightmaps and segmentation masks."""
    # Capture near-orthographic RGB-D images and segmentation masks.
    color, depth, segm = env.render_camera(oracle_cams[0])

    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = utils.reconstruct_heightmaps(
        [color], [depth], oracle_cams, bounds, pix_size)

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
    return cmap, hmap, mask


def get_random_pose(env, obj_size):
    """Get random collision-free object pose within workspace bounds."""

    # Get erosion size of object in pixels.
    max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
    erode_size = int(np.round(max_size / pix_size))

    _, hmap, obj_mask = get_true_image(env)

    # Randomly sample an object pose within free-space pixels.
    free = np.ones(obj_mask.shape, dtype=np.uint8)
    for obj_ids in env.obj_ids.values():
        for obj_id in obj_ids:
            free[obj_mask == obj_id] = 0
    free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
    free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
    if np.sum(free) == 0:
        return None, None
    pix = utils.sample_distribution(np.float32(free))
    pos = utils.pix_to_xyz(pix, hmap, bounds, pix_size)
    pos = (pos[0], pos[1], obj_size[2] / 2)
    theta = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
    return pos, rot

def setup_environment(object_list, disp=False):
    env = Environment(
        './environments/assets/',
        disp=disp,
        shared_memory=False,
        hz=480,
        # record_cfg=cfg['record']
    )
    env.seed(0)
    task = GeneralTask()
    env.set_task(task)
    obs = env.reset()
    lang_objects = object_list
    colors = []

    n_bowls = 0
    bowl_size = (0.12, 0.12, 0)
    bowl_urdf = 'bowl/bowl.urdf'

    n_blocks = 0
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'stacking/block.urdf'
    objects = {}

    for obj in lang_objects:
        print(obj)
        color_name, item = obj.split(' ')
        color = utils.COLORS[color_name]
        colors.append(color)
        if item == 'block':
            n_blocks += 1
            block_pose = get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=color + [1])
            objects[obj] = block_id
        if item == 'bowl':
            n_bowls += 1
            bowl_pose = get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose)
            p.changeVisualShape(bowl_id, -1, rgbaColor=color + [1])
            objects[obj] = bowl_id
        input("Enter to continue")
    return env, task, obs, objects

def actor(lang_goals, object_list = None, env = None, task = None, obs = None, obj_info = None, disp = False):
    if object_list and env == None:
        env, task, obs, obj_info = setup_environment(object_list, disp=disp)
    agent = task.oracle(env)
    for line in lang_goals.splitlines():
        if 'place(' in line:
            prompt = line.strip()
            print(prompt)
            task.set_goals(env, prompt, obj_info=obj_info)
            act = agent.act(obs, env.info)
            obs, _, done, info = env.step(act)
            input("Enter to continue")
    return env, task, obs, obj_info