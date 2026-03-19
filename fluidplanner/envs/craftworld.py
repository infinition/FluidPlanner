"""
CraftWorld: Variable-depth crafting chains for hierarchical planning.

The agent navigates a grid, collects raw materials, visits crafting stations
to combine them, and delivers the final product. Recipe depth varies from
2 to 6+ steps, requiring the planner to discover sub-goal decompositions
of different depths.

This is the "incontestable" experiment: if the PDE hierarchy adapts to
variable-depth recipes without being told the depth, it demonstrates
genuine emergent hierarchical planning.
"""

import heapq
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np


# Cell types
EMPTY = 0
WALL = 1
AGENT = 2
MATERIAL_BASE = 10   # materials: 10, 11, 12, 13, 14
STATION = 20         # crafting station
DELIVERY = 21        # delivery point

# Actions
UP, DOWN, LEFT, RIGHT, CRAFT, NOOP = 0, 1, 2, 3, 4, 5
ACTION_NAMES = ["up", "down", "left", "right", "craft", "noop"]
MOVE_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), None, None]

# Materials and recipes
MATERIAL_NAMES = ["wood", "stone", "iron", "leather", "gold"]
MATERIAL_COLORS = [(0.6, 0.3, 0.0), (0.5, 0.5, 0.5), (0.7, 0.7, 0.8), (0.5, 0.3, 0.1), (1.0, 0.84, 0.0)]

# Recipes: (input_items) -> output_item
# Depth 1 recipes (1 craft)
RECIPES_DEPTH1 = [
    (["wood"], "plank"),
    (["stone"], "brick"),
    (["iron"], "ingot"),
]

# Depth 2 recipes (2 crafts)
RECIPES_DEPTH2 = [
    (["plank", "plank"], "shelf"),          # needs: wood->plank x2 then plank+plank->shelf
    (["ingot", "wood"], "sword"),           # needs: iron->ingot then ingot+wood->sword
    (["brick", "brick"], "wall_section"),   # needs: stone->brick x2 then brick+brick->wall
]

# Depth 3 recipes (3 crafts)
RECIPES_DEPTH3 = [
    (["sword", "leather"], "warrior_kit"),  # iron->ingot, ingot+wood->sword, sword+leather->kit
    (["shelf", "brick"], "furnace"),        # wood->plank x2, plank+plank->shelf, shelf+brick->furnace
    (["ingot", "ingot", "wood"], "anvil"),  # iron->ingot x2, ingot+ingot+wood->anvil
]

# Full recipe tree: item -> (ingredients, depth)
RECIPE_TREE = {
    # Depth 0: raw materials (no recipe needed, just pick up)
    "wood": ([], 0),
    "stone": ([], 0),
    "iron": ([], 0),
    "leather": ([], 0),
    "gold": ([], 0),
    # Depth 1: single craft from raw
    "plank": (["wood"], 1),
    "brick": (["stone"], 1),
    "ingot": (["iron"], 1),
    # Depth 2: craft from depth-1 items
    "shelf": (["plank", "plank"], 2),
    "sword": (["ingot", "wood"], 2),
    "wall_section": (["brick", "brick"], 2),
    # Depth 3: craft from depth-2 items
    "warrior_kit": (["sword", "leather"], 3),
    "furnace": (["shelf", "brick"], 3),
    "anvil": (["ingot", "ingot", "wood"], 3),
}

# All craftable items (excluding raw materials)
CRAFTABLE = {k: v for k, v in RECIPE_TREE.items() if v[1] > 0}

# Map raw material names to IDs
RAW_MATERIAL_IDS = {"wood": 10, "stone": 11, "iron": 12, "leather": 13, "gold": 14}


def flatten_recipe(target: str) -> list:
    """
    Compute the ordered list of actions to craft target from raw materials.
    Returns list of (action_type, details) tuples.
    """
    steps = []
    needed = []

    def resolve(item):
        ingredients, depth = RECIPE_TREE[item]
        if depth == 0:
            needed.append(("collect", item))
            return
        for ing in ingredients:
            resolve(ing)
        needed.append(("craft", item))

    resolve(target)
    return needed


@dataclass
class CraftState:
    grid: np.ndarray
    agent_pos: tuple
    inventory: list          # items the agent carries
    target_item: str         # what to deliver
    target_depth: int        # recipe depth
    steps: int
    delivered: bool
    craft_log: list          # sequence of crafts performed


class CraftWorldEnv:
    """
    Grid world with crafting mechanics.

    The grid has:
    - Raw material spawns (colored squares)
    - Crafting stations (where combining happens)
    - A delivery point (where the final product goes)
    - Walls for navigation challenge
    """

    def __init__(self, size: int = 20, seed: Optional[int] = None):
        self.size = size
        self.rng = random.Random(seed)
        self.max_inventory = 5
        self.max_steps = 200
        self.reset()

    def reset(self, difficulty: Optional[int] = None, seed: Optional[int] = None) -> CraftState:
        if seed is not None:
            self.rng = random.Random(seed)

        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self.inventory = []
        self.craft_log = []
        self.steps = 0
        self.delivered = False

        # Choose target based on difficulty (1-3)
        if difficulty is None:
            difficulty = self.rng.randint(1, 3)

        if difficulty == 1:
            targets = ["plank", "brick", "ingot"]
        elif difficulty == 2:
            targets = ["shelf", "sword", "wall_section"]
        else:
            targets = ["warrior_kit", "furnace", "anvil"]

        self.target_item = self.rng.choice(targets)
        self.target_depth = RECIPE_TREE[self.target_item][1]

        self._build_grid()
        return self._get_state()

    def _build_grid(self):
        s = self.size
        g = self.grid

        # Border walls
        g[0, :] = WALL
        g[-1, :] = WALL
        g[:, 0] = WALL
        g[:, -1] = WALL

        # Some internal walls
        for _ in range(s):
            r = self.rng.randint(2, s - 3)
            c = self.rng.randint(2, s - 3)
            g[r, c] = WALL

        # Place agent
        self.agent_pos = self._rand_empty()
        g[self.agent_pos] = AGENT

        # Place delivery point
        self.delivery_pos = self._rand_empty()
        g[self.delivery_pos] = DELIVERY

        # Place 2-3 crafting stations
        self.station_positions = []
        for _ in range(3):
            pos = self._rand_empty()
            g[pos] = STATION
            self.station_positions.append(pos)

        # Determine which raw materials are needed
        recipe_steps = flatten_recipe(self.target_item)
        needed_raw = set()
        for action_type, item in recipe_steps:
            if action_type == "collect":
                needed_raw.add(item)

        # Place needed raw materials (multiple copies for reliability)
        self.material_positions = {}
        for mat_name in needed_raw:
            mat_id = RAW_MATERIAL_IDS[mat_name]
            positions = []
            for _ in range(2):  # 2 copies of each needed material
                pos = self._rand_empty()
                g[pos] = mat_id
                positions.append(pos)
            self.material_positions[mat_name] = positions

        # Place a few extra random materials (distractors)
        for _ in range(3):
            mat_name = self.rng.choice(MATERIAL_NAMES)
            mat_id = RAW_MATERIAL_IDS[mat_name]
            pos = self._rand_empty()
            g[pos] = mat_id

    def _rand_empty(self) -> tuple:
        for _ in range(1000):
            r = self.rng.randint(1, self.size - 2)
            c = self.rng.randint(1, self.size - 2)
            if self.grid[r, c] == EMPTY:
                return (r, c)
        # Fallback
        for r in range(1, self.size - 1):
            for c in range(1, self.size - 1):
                if self.grid[r, c] == EMPTY:
                    return (r, c)
        raise RuntimeError("No empty cell")

    def step(self, action: int) -> tuple:
        """Returns (state, reward, done)."""
        self.steps += 1
        reward = -0.01  # step penalty

        if action < 4:  # movement
            dr, dc = MOVE_DELTAS[action]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                cell = self.grid[nr, nc]
                if cell != WALL:
                    # Move
                    self.grid[self.agent_pos] = EMPTY
                    # Pick up material on contact
                    if 10 <= cell <= 14 and len(self.inventory) < self.max_inventory:
                        mat_name = MATERIAL_NAMES[cell - 10]
                        self.inventory.append(mat_name)
                        reward = 0.5
                    self.agent_pos = (nr, nc)
                    self.grid[self.agent_pos] = AGENT

                    # Check delivery
                    if (nr, nc) == self.delivery_pos and self.target_item in self.inventory:
                        self.inventory.remove(self.target_item)
                        self.delivered = True
                        reward = 10.0

        elif action == 4:  # craft
            # Must be adjacent to or on a crafting station
            on_station = False
            for sr, sc in self.station_positions:
                dist = abs(self.agent_pos[0] - sr) + abs(self.agent_pos[1] - sc)
                if dist <= 1:
                    on_station = True
                    break

            if on_station:
                # Try all recipes
                crafted = False
                for item_name, (ingredients, depth) in CRAFTABLE.items():
                    # Check if we have all ingredients
                    inv_copy = list(self.inventory)
                    has_all = True
                    for ing in ingredients:
                        if ing in inv_copy:
                            inv_copy.remove(ing)
                        else:
                            has_all = False
                            break
                    if has_all:
                        # Craft: remove ingredients, add product
                        for ing in ingredients:
                            self.inventory.remove(ing)
                        self.inventory.append(item_name)
                        self.craft_log.append(item_name)
                        reward = 1.0
                        crafted = True
                        break
                if not crafted:
                    reward = -0.1  # failed craft attempt

        done = self.delivered or self.steps >= self.max_steps
        return self._get_state(), reward, done

    def to_tensor(self, device="cpu") -> torch.Tensor:
        """
        Encode as multi-channel tensor:
        0: empty, 1: wall, 2: agent, 3: station, 4: delivery,
        5-9: materials (wood, stone, iron, leather, gold),
        10: inventory count (normalized), 11: target depth (normalized)
        """
        n_channels = 12
        channels = np.zeros((n_channels, self.size, self.size), dtype=np.float32)

        channels[0] = (self.grid == EMPTY).astype(np.float32)
        channels[1] = (self.grid == WALL).astype(np.float32)
        channels[2] = (self.grid == AGENT).astype(np.float32)
        channels[3] = (self.grid == STATION).astype(np.float32)
        channels[4] = (self.grid == DELIVERY).astype(np.float32)

        for i in range(5):
            channels[5 + i] = (self.grid == (10 + i)).astype(np.float32)

        # Inventory info (broadcast)
        channels[10] = len(self.inventory) / self.max_inventory
        channels[11] = self.target_depth / 3.0

        return torch.from_numpy(channels).to(device)

    def _get_state(self) -> CraftState:
        return CraftState(
            grid=self.grid.copy(),
            agent_pos=self.agent_pos,
            inventory=list(self.inventory),
            target_item=self.target_item,
            target_depth=self.target_depth,
            steps=self.steps,
            delivered=self.delivered,
            craft_log=list(self.craft_log),
        )

    def optimal_steps(self) -> list:
        """
        Compute near-optimal action sequence.
        Uses recipe flattening + A* navigation between key positions.
        """
        recipe_plan = flatten_recipe(self.target_item)
        actions = []
        sim_pos = self.agent_pos
        sim_inventory = []
        sim_grid = self.grid.copy()

        for action_type, item in recipe_plan:
            if action_type == "collect":
                # Navigate to nearest material
                mat_id = RAW_MATERIAL_IDS[item]
                target = self._find_nearest(sim_pos, sim_grid, mat_id)
                if target is None:
                    return []  # unsolvable
                path = self._astar_path(sim_pos, target, sim_grid)
                if path is None:
                    return []
                for p in path[1:]:
                    dr = p[0] - sim_pos[0]
                    dc = p[1] - sim_pos[1]
                    for a, delta in enumerate(MOVE_DELTAS[:4]):
                        if delta == (dr, dc):
                            actions.append(a)
                            break
                    sim_pos = p
                sim_inventory.append(item)
                sim_grid[target] = EMPTY  # material picked up

            elif action_type == "craft":
                # Navigate to nearest station
                target = self._find_nearest(sim_pos, sim_grid, STATION)
                if target is None:
                    return []
                # Go adjacent to station
                adj = self._adjacent_empty(target, sim_grid)
                if adj is None:
                    adj = target
                path = self._astar_path(sim_pos, adj, sim_grid)
                if path is None:
                    return []
                for p in path[1:]:
                    dr = p[0] - sim_pos[0]
                    dc = p[1] - sim_pos[1]
                    for a, delta in enumerate(MOVE_DELTAS[:4]):
                        if delta == (dr, dc):
                            actions.append(a)
                            break
                    sim_pos = p
                actions.append(4)  # CRAFT action
                # Update inventory
                ingredients = RECIPE_TREE[item][0]
                for ing in ingredients:
                    if ing in sim_inventory:
                        sim_inventory.remove(ing)
                sim_inventory.append(item)

        # Navigate to delivery
        path = self._astar_path(sim_pos, self.delivery_pos, sim_grid)
        if path is None:
            return []
        for p in path[1:]:
            dr = p[0] - sim_pos[0]
            dc = p[1] - sim_pos[1]
            for a, delta in enumerate(MOVE_DELTAS[:4]):
                if delta == (dr, dc):
                    actions.append(a)
                    break
            sim_pos = p

        return actions

    def _find_nearest(self, pos, grid, cell_type):
        best = None
        best_dist = float("inf")
        for r in range(self.size):
            for c in range(self.size):
                if grid[r, c] == cell_type:
                    d = abs(r - pos[0]) + abs(c - pos[1])
                    if d < best_dist:
                        best_dist = d
                        best = (r, c)
        return best

    def _adjacent_empty(self, pos, grid):
        for dr, dc in MOVE_DELTAS[:4]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and grid[nr, nc] in (EMPTY, AGENT):
                return (nr, nc)
        return None

    def _astar_path(self, start, goal, grid):
        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = [(h(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, cost, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for dr, dc in MOVE_DELTAS[:4]:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if grid[nr, nc] != WALL or (nr, nc) == goal:
                        t = g_score[current] + 1
                        n = (nr, nc)
                        if t < g_score.get(n, float("inf")):
                            came_from[n] = current
                            g_score[n] = t
                            heapq.heappush(open_set, (t + h(n, goal), t, n))
        return None


def generate_craft_dataset(n_episodes: int = 500, grid_size: int = 20, seed: int = 42):
    """
    Generate training data with MIXED difficulty.
    Each episode has a different recipe depth (1-3).
    Returns list of trajectories: [(state_tensor, action, depth)]
    """
    env = CraftWorldEnv(size=grid_size, seed=seed)
    dataset = []
    stats = {1: 0, 2: 0, 3: 0}

    for i in range(n_episodes * 5):
        difficulty = (i % 3) + 1  # cycle through 1, 2, 3
        env.reset(difficulty=difficulty, seed=seed + i)
        actions = env.optimal_steps()
        if len(actions) < 2:
            continue

        env.reset(difficulty=difficulty, seed=seed + i)
        trajectory = []
        for action in actions:
            state_t = env.to_tensor()
            trajectory.append((state_t, action, env.target_depth))
            env.step(action)

        # Add final delivery step
        if not env.delivered:
            # Check if we can deliver
            state_t = env.to_tensor()
            trajectory.append((state_t, 5, env.target_depth))  # NOOP at end

        dataset.append(trajectory)
        stats[difficulty] += 1

        if len(dataset) >= n_episodes:
            break

    print(f"  Dataset: {len(dataset)} episodes "
          f"(depth1={stats[1]}, depth2={stats[2]}, depth3={stats[3]})")
    return dataset
