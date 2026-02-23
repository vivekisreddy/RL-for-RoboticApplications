import gymnasium as gym
import pickle
import numpy as np
from PIL import Image
import cv2
import sys

ENV_NAME = "CliffWalking-v1"
CELL_SIZE = 50
GRID_ROWS = 4
GRID_COLS = 12

def initialize_frame():
    img = np.ones((GRID_ROWS*CELL_SIZE, GRID_COLS*CELL_SIZE,3), dtype=np.uint8)*255
    for i in range(GRID_COLS+1):
        cv2.line(img, (i*CELL_SIZE,0), (i*CELL_SIZE, GRID_ROWS*CELL_SIZE), (180,180,180),1)
    for i in range(GRID_ROWS+1):
        cv2.line(img, (0,i*CELL_SIZE), (GRID_COLS*CELL_SIZE, i*CELL_SIZE), (180,180,180),1)
    for c in range(1,11):
        cv2.rectangle(img, (c*CELL_SIZE,3*CELL_SIZE), ((c+1)*CELL_SIZE,4*CELL_SIZE), (150,0,150), -1)
    cv2.putText(img, "S", (10, 4*CELL_SIZE-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
    cv2.putText(img, "G", (11*CELL_SIZE+10, 4*CELL_SIZE-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
    return img

def draw_agent(img, state, color):
    row, col = np.unravel_index(state, (GRID_ROWS, GRID_COLS))
    center = (col*CELL_SIZE + CELL_SIZE//2, row*CELL_SIZE + CELL_SIZE//2)
    cv2.circle(img, center, 12, color, -1)
    return img

def draw_path(img, state, color):
    row, col = np.unravel_index(state, (GRID_ROWS, GRID_COLS))
    center = (col*CELL_SIZE + CELL_SIZE//2, row*CELL_SIZE + CELL_SIZE//2)
    cv2.circle(img, center, 6, color, -1)
    return img

def generate_gif(method):
    env = gym.make(ENV_NAME)
    q_table = pickle.load(open(f"{method}_q_table.pkl","rb"))
    frames = []

    state, _ = env.reset()
    state = int(state)
    done = False
    visited = []

    if method=="sarsa":
        agent_color=(255,0,0)
        path_color=(255,0,0)
        filename="sarsa_policy.gif"
    else:
        agent_color=(0,0,255)
        path_color=(0,0,255)
        filename="q_learning_policy.gif"

    max_steps = 100
    steps = 0
    while not done and steps < max_steps:
        frame = initialize_frame()
        for s in visited:
            frame = draw_path(frame, s, path_color)
        frame = draw_agent(frame, state, agent_color)
        frames.append(Image.fromarray(frame))

        action = np.argmax(q_table[state])
        visited.append(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        state = int(next_state)
        done = terminated or truncated
        steps += 1

    # Final frame
    frame = initialize_frame()
    for s in visited:
        frame = draw_path(frame, s, path_color)
    frames.append(Image.fromarray(frame))

    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=300, loop=0)
    print(f"{method.upper()} GIF saved as {filename}")

if __name__=="__main__":
    method = sys.argv[1]
    generate_gif(method)