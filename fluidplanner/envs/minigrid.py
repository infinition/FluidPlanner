"""
MiniGrid environment for hierarchical planning experiments.

Grid world with walls, keys, doors, and goals.
Agent must find key -> unlock door -> reach goal.
Provides optimal trajectories via A* for imitation learning.
"""

import heapq
import random
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np


# Cell types
EMPTY = 0
WALL = 1
KEY = 2
DOOR = 3
GOAL = 4
AGENT = 5

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTION_NAMES = ["up", "down", "left", "right"]
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class GridState:
    """Current state of the grid environment."""
    grid: np.ndarray        # (H, W) cell types
    agent_pos: tuple        # (row, col)
    has_key: bool           # whether agent picked up key
    door_open: bool         # whether door is unlocked
    goal_reached: bool
    steps: int


class MiniGridEnv:
    """
    Grid world with key-door-goal mechanics.

    The agent must:
    1. Navigate to the key (K)
    2. Pick it up (automatic on contact)
    3. Navigate to the door (D) to open it
    4. Navigate to the goal (G)

    Walls block movement. The door blocks movement until opened with key.
    """

    def __init__(self, size: int = 16, seed: Optional[int] = None):
        self.size = size
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.reset()

    def reset(self, maze_id: Optional[int] = None) -> GridState:
        """Generate a new random maze with key, door, goal."""
        if maze_id is not None:
            self.rng = random.Random(maze_id)
            self.np_rng = np.random.RandomState(maze_id)

        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self._generate_maze()
        self.has_key = False
        self.door_open = False
        self.goal_reached = False
        self.steps = 0
        return self._get_state()

    def _generate_maze(self):
        """Generate a maze with rooms, corridors, key, door, goal."""
        s = self.size
        g = self.grid

        # Border walls
        g[0, :] = WALL
        g[-1, :] = WALL
        g[:, 0] = WALL
        g[:, -1] = WALL

        # Vertical divider wall with gap (creates 2 rooms)
        mid_col = s // 2
        g[:, mid_col] = WALL

        # Door in the divider wall
        door_row = self.rng.randint(2, s - 3)
        g[door_row, mid_col] = DOOR
        self.door_pos = (door_row, mid_col)

        # Add some random walls in each room for complexity
        for _ in range(s // 2):
            r = self.rng.randint(1, s - 2)
            c = self.rng.randint(1, mid_col - 1)
            if g[r, c] == EMPTY:
                g[r, c] = WALL

        for _ in range(s // 2):
            r = self.rng.randint(1, s - 2)
            c = self.rng.randint(mid_col + 1, s - 2)
            if g[r, c] == EMPTY:
                g[r, c] = WALL

        # Place agent in left room
        self.agent_pos = self._random_empty(1, s - 2, 1, mid_col - 1)
        g[self.agent_pos] = AGENT

        # Place key in left room (different from agent)
        self.key_pos = self._random_empty(1, s - 2, 1, mid_col - 1)
        g[self.key_pos] = KEY

        # Place goal in right room
        self.goal_pos = self._random_empty(1, s - 2, mid_col + 1, s - 2)
        g[self.goal_pos] = GOAL

    def _random_empty(self, r_min, r_max, c_min, c_max) -> tuple:
        """Find a random empty cell in the given range."""
        for _ in range(1000):
            r = self.rng.randint(r_min, r_max)
            c = self.rng.randint(c_min, c_max)
            if self.grid[r, c] == EMPTY:
                return (r, c)
        # Fallback: find any empty cell
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if self.grid[r, c] == EMPTY:
                    return (r, c)
        raise RuntimeError("No empty cell found")

    def step(self, action: int) -> tuple:
        """Execute action, return (state, reward, done)."""
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # Bounds check
        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
            self.steps += 1
            return self._get_state(), -0.1, False

        cell = self.grid[nr, nc]

        # Wall blocks
        if cell == WALL:
            self.steps += 1
            return self._get_state(), -0.1, False

        # Door blocks unless has key
        if cell == DOOR and not self.has_key:
            self.steps += 1
            return self._get_state(), -0.1, False

        # Move agent
        self.grid[self.agent_pos] = EMPTY
        self.agent_pos = (nr, nc)

        reward = -0.01  # step cost

        # Pick up key
        if cell == KEY:
            self.has_key = True
            reward = 1.0

        # Open door
        if cell == DOOR and self.has_key:
            self.door_open = True
            self.grid[nr, nc] = EMPTY
            reward = 1.0

        # Reach goal
        if cell == GOAL:
            self.goal_reached = True
            reward = 10.0

        self.grid[self.agent_pos] = AGENT
        self.steps += 1
        return self._get_state(), reward, self.goal_reached

    def _get_state(self) -> GridState:
        return GridState(
            grid=self.grid.copy(),
            agent_pos=self.agent_pos,
            has_key=self.has_key,
            door_open=self.door_open,
            goal_reached=self.goal_reached,
            steps=self.steps,
        )

    def to_tensor(self, device="cpu") -> torch.Tensor:
        """Encode grid as multi-channel tensor (6, H, W)."""
        channels = np.zeros((6, self.size, self.size), dtype=np.float32)
        for cell_type in range(6):
            channels[cell_type] = (self.grid == cell_type).astype(np.float32)
        # Add key-held channel (broadcast over whole grid)
        if self.has_key:
            channels[2] *= 0.5  # dim key channel if held
            channels[5] += 0.5  # brighten agent
        return torch.from_numpy(channels).to(device)

    def optimal_trajectory(self) -> list:
        """Compute optimal action sequence via A* with key-door logic."""
        # Phase 1: agent -> key
        path_to_key = self._astar(self.agent_pos, self.key_pos, has_key=False)
        if path_to_key is None:
            return []

        # Phase 2: key -> door
        path_to_door = self._astar(self.key_pos, self.door_pos, has_key=True)
        if path_to_door is None:
            return []

        # Phase 3: door -> goal (door is now open)
        path_to_goal = self._astar(self.door_pos, self.goal_pos, has_key=True)
        if path_to_goal is None:
            return []

        # Convert position sequences to actions
        full_path = path_to_key + path_to_door[1:] + path_to_goal[1:]
        actions = []
        for i in range(len(full_path) - 1):
            dr = full_path[i + 1][0] - full_path[i][0]
            dc = full_path[i + 1][1] - full_path[i][1]
            for a, (adr, adc) in enumerate(ACTION_DELTAS):
                if dr == adr and dc == adc:
                    actions.append(a)
                    break
        return actions

    def _astar(self, start, goal, has_key=False) -> list:
        """A* pathfinding on the grid."""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = [(heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, cost, current = heapq.heappop(open_set)
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for dr, dc in ACTION_DELTAS:
                nr, nc = current[0] + dr, current[1] + dc
                if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
                    continue
                cell = self.grid[nr, nc]
                if cell == WALL:
                    continue
                if cell == DOOR and not has_key:
                    continue

                tentative = g_score[current] + 1
                neighbor = (nr, nc)
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    heapq.heappush(open_set, (tentative + heuristic(neighbor, goal), tentative, neighbor))

        return None


def generate_dataset(n_mazes: int = 1000, grid_size: int = 16, seed: int = 42):
    """Generate training dataset of (grid_tensor, optimal_actions) pairs."""
    env = MiniGridEnv(size=grid_size, seed=seed)
    dataset = []
    valid = 0
    for i in range(n_mazes * 3):  # oversample to get enough solvable mazes
        env.reset(maze_id=seed + i)
        actions = env.optimal_trajectory()
        if len(actions) < 3:
            continue
        # Store full trajectory: list of (state_tensor, action, phase)
        trajectory = []
        for step_idx, action in enumerate(actions):
            state_t = env.to_tensor()
            # Phase: 0=going to key, 1=going to door, 2=going to goal
            if not env.has_key:
                phase = 0
            elif not env.door_open:
                phase = 1
            else:
                phase = 2
            trajectory.append((state_t, action, phase))
            env.step(action)
        dataset.append(trajectory)
        valid += 1
        if valid >= n_mazes:
            break
    return dataset
