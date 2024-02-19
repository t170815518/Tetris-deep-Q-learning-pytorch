"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    # initialization
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)

    # 训练开始
    epoch = 0
    while epoch < opt.num_epochs:
        # 获取环境的下一个可能状态
        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        # 获取下一步的动作和状态
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        # 如果可以使用CUDA，则将next_states转移到GPU
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        # 在不需要计算梯度的情况下，使用模型预测下一个状态的Q值
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]

        model.train()
        # 如果是随机动作，则从可能的动作中随机选择一个；否则，选择具有最高预测Q值的动作
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        # 获取选择的下一个状态和动作
        next_state = next_states[index, :]
        action = next_actions[index]

        # 执行动作并获取奖励和完成状态
        reward, done = env.step(action, render=True)

        if torch.cuda.is_available():
            next_state = next_state.cuda()

        # 将当前状态、奖励、下一个状态和完成状态添加到重播内存中
        replay_memory.append([state, reward, next_state, done])

        # 如果游戏结束，则重置状态并记录最终得分、Tetromino数量和清除的行数；否则，将状态更新为下一个状态
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue

        # 如果重播内存的长度小于重播内存大小的10%，则继续收集经验，不进行学习
        if len(replay_memory) < opt.replay_memory_size / 10:
            print(f'len(replay_memory)={len(replay_memory)}<{opt.replay_memory_size / 10}\t')
            continue

        # 从重播内存中随机抽取一批样本进行学习
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # 计算当前状态的Q值
        q_values = model(state_batch)

        # 在不需要计算梯度的情况下，预测下一个状态的Q值
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        # 计算目标Q值
        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        # 计算损失并进行反向传播
        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        # 将训练信息写入TensorBoard
        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        # 每隔一定的周期保存模型
        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/tetris".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
