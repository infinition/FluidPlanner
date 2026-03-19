"""
Visualization server for FluidPlanner.

Serves an interactive HTML dashboard showing:
- Grid world with agent, key, door, goal
- PDE field activations at levels 0, 1, 2
- Agent's planned trajectory vs optimal
- Real-time stepping through episodes

Usage: python experiments/serve_viz.py --checkpoint checkpoints/best_fluid.pt
"""

import argparse
import json
import os
import sys
import http.server
import socketserver
import threading

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fluidplanner.envs.minigrid import MiniGridEnv, EMPTY, WALL, KEY, DOOR, GOAL, AGENT, ACTION_NAMES
from fluidplanner.planner import FluidPlanner


def generate_episode_data(model, env, maze_id, device):
    """Run one episode and collect all data for visualization."""
    env.reset(maze_id=maze_id)
    optimal_actions = env.optimal_trajectory()
    env.reset(maze_id=maze_id)

    frames = []
    model.eval()

    max_steps = max(len(optimal_actions) * 3, 30)

    for step in range(max_steps):
        state = env.to_tensor(device).unsqueeze(0)

        with torch.no_grad():
            out = model(state, return_fields=True)

        action = out["action_logits"].argmax(dim=-1).item()
        action_probs = torch.softmax(out["action_logits"], dim=-1).squeeze().cpu().tolist()
        phase_probs = torch.softmax(out["phase_logits"], dim=-1).squeeze().cpu().tolist()

        # Extract PCA-reduced field visualizations (3 channels -> RGB)
        fields_rgb = {}
        for name, field in out.get("fields", {}).items():
            f = field.squeeze(0)  # (C, H, W)
            # Take first 3 channels as RGB proxy
            rgb = f[:3].cpu().numpy()  # (3, H, W)
            # Normalize to [0, 1]
            for c in range(3):
                vmin, vmax = rgb[c].min(), rgb[c].max()
                if vmax - vmin > 1e-6:
                    rgb[c] = (rgb[c] - vmin) / (vmax - vmin)
                else:
                    rgb[c] = 0.5
            fields_rgb[name] = rgb.tolist()

        frame = {
            "step": step,
            "grid": env.grid.tolist(),
            "agent_pos": list(env.agent_pos),
            "has_key": env.has_key,
            "door_open": env.door_open,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "action_probs": action_probs,
            "phase_probs": phase_probs,
            "fields": fields_rgb,
            "goal_reached": env.goal_reached,
        }
        frames.append(frame)

        if env.goal_reached:
            break

        env.step(action)

    # Also record optimal trajectory
    env.reset(maze_id=maze_id)
    optimal_positions = [list(env.agent_pos)]
    for a in optimal_actions:
        env.step(a)
        optimal_positions.append(list(env.agent_pos))

    return {
        "maze_id": maze_id,
        "grid_size": env.size,
        "frames": frames,
        "optimal_path": optimal_positions,
        "optimal_length": len(optimal_actions),
        "model_length": len(frames),
        "solved": frames[-1]["goal_reached"] if frames else False,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FluidPlanner Visualization</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0f1419; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
.header { background: #1a1f2e; padding: 12px 24px; border-bottom: 1px solid #2a3040; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 18px; color: #c9956b; }
.header .status { font-size: 13px; color: #8892a0; }
.controls { background: #1a1f2e; padding: 10px 24px; border-bottom: 1px solid #2a3040; display: flex; gap: 12px; align-items: center; }
.controls button { background: #2a3545; color: #e0e0e0; border: 1px solid #3a4555; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 13px; }
.controls button:hover { background: #3a4555; }
.controls button.active { background: #c9956b; color: #000; border-color: #c9956b; }
.controls input[type=range] { width: 200px; }
.controls select { background: #2a3545; color: #e0e0e0; border: 1px solid #3a4555; padding: 4px 8px; border-radius: 4px; }
.main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px 24px; }
.panel { background: #1a1f2e; border: 1px solid #2a3040; border-radius: 6px; padding: 12px; }
.panel h3 { font-size: 13px; color: #c9956b; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
canvas { display: block; margin: 0 auto; image-rendering: pixelated; border: 1px solid #2a3040; }
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 12px; }
.info-item { display: flex; justify-content: space-between; padding: 3px 6px; background: #0f1419; border-radius: 3px; }
.info-label { color: #8892a0; }
.info-value { color: #e0e0e0; font-weight: 600; }
.bar-chart { display: flex; gap: 4px; align-items: flex-end; height: 60px; margin-top: 8px; }
.bar { flex: 1; background: #c9956b; border-radius: 2px 2px 0 0; transition: height 0.2s; position: relative; }
.bar-label { font-size: 9px; text-align: center; color: #8892a0; margin-top: 2px; }
.fields-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
.field-panel { text-align: center; }
.field-panel canvas { margin: 4px auto; }
.field-label { font-size: 11px; color: #8892a0; }
.solved { color: #4ade80; font-weight: 700; }
.failed { color: #f87171; font-weight: 700; }
.phase-indicator { display: flex; gap: 6px; margin-top: 6px; }
.phase { padding: 3px 10px; border-radius: 12px; font-size: 11px; background: #2a3545; }
.phase.active { background: #c9956b; color: #000; font-weight: 600; }
</style>
</head>
<body>
<div class="header">
  <h1>FluidPlanner</h1>
  <div class="status" id="statusText">Loading...</div>
</div>
<div class="controls">
  <button id="btnPrev" onclick="prevStep()">Prev</button>
  <button id="btnPlay" onclick="togglePlay()">Play</button>
  <button id="btnNext" onclick="nextStep()">Next</button>
  <input type="range" id="stepSlider" min="0" max="0" value="0" oninput="setStep(this.value)">
  <span id="stepLabel" style="font-size:12px;min-width:80px;">Step 0/0</span>
  <select id="mazeSelect" onchange="loadMaze(this.value)"></select>
  <button onclick="randomMaze()">Random</button>
</div>
<div class="main">
  <div class="panel">
    <h3>Grid World</h3>
    <canvas id="gridCanvas" width="512" height="512"></canvas>
    <div class="info-grid" style="margin-top:8px;">
      <div class="info-item"><span class="info-label">Action</span><span class="info-value" id="infoAction">-</span></div>
      <div class="info-item"><span class="info-label">Has Key</span><span class="info-value" id="infoKey">No</span></div>
      <div class="info-item"><span class="info-label">Door</span><span class="info-value" id="infoDoor">Closed</span></div>
      <div class="info-item"><span class="info-label">Result</span><span class="info-value" id="infoResult">-</span></div>
    </div>
    <div style="margin-top:8px;">
      <span style="font-size:11px;color:#8892a0;">Action Probabilities</span>
      <div class="bar-chart" id="actionBars"></div>
      <div style="display:flex;gap:4px;justify-content:space-around;">
        <span class="bar-label">Up</span><span class="bar-label">Down</span><span class="bar-label">Left</span><span class="bar-label">Right</span>
      </div>
    </div>
    <div style="margin-top:8px;">
      <span style="font-size:11px;color:#8892a0;">Detected Phase</span>
      <div class="phase-indicator" id="phaseIndicator">
        <span class="phase" id="phase0">Key</span>
        <span class="phase" id="phase1">Door</span>
        <span class="phase" id="phase2">Goal</span>
      </div>
    </div>
  </div>
  <div class="panel">
    <h3>PDE Field Activations</h3>
    <div class="fields-row">
      <div class="field-panel">
        <div class="field-label">Level 2 (Slow - Global Plan)</div>
        <canvas id="field2" width="256" height="256"></canvas>
      </div>
      <div class="field-panel">
        <div class="field-label">Level 1 (Medium - Sub-goals)</div>
        <canvas id="field1" width="256" height="256"></canvas>
      </div>
      <div class="field-panel">
        <div class="field-label">Level 0 (Fast - Actions)</div>
        <canvas id="field0" width="256" height="256"></canvas>
      </div>
    </div>
    <div style="margin-top:12px;">
      <h3>Trajectory Comparison</h3>
      <canvas id="trajCanvas" width="512" height="200"></canvas>
      <div class="info-grid" style="margin-top:6px;">
        <div class="info-item"><span class="info-label">Optimal steps</span><span class="info-value" id="infoOptimal">-</span></div>
        <div class="info-item"><span class="info-label">Model steps</span><span class="info-value" id="infoModel">-</span></div>
      </div>
    </div>
  </div>
</div>

<script>
let episodes = {};
let currentMaze = null;
let currentStep = 0;
let playing = false;
let playTimer = null;

const CELL_COLORS = {
  0: '#1a1f2e',   // empty
  1: '#4a5568',   // wall
  2: '#fbbf24',   // key
  3: '#a855f7',   // door
  4: '#4ade80',   // goal
  5: '#ef4444',   // agent
};

async function init() {
  const resp = await fetch('/api/episodes');
  episodes = await resp.json();
  const select = document.getElementById('mazeSelect');
  Object.keys(episodes).forEach(id => {
    const opt = document.createElement('option');
    opt.value = id;
    const ep = episodes[id];
    opt.textContent = `Maze ${id} (${ep.solved ? 'Solved' : 'Failed'} in ${ep.model_length} steps)`;
    select.appendChild(opt);
  });
  if (Object.keys(episodes).length > 0) {
    loadMaze(Object.keys(episodes)[0]);
  }
  document.getElementById('statusText').textContent =
    `${Object.keys(episodes).length} episodes loaded`;
}

function loadMaze(id) {
  currentMaze = episodes[id];
  currentStep = 0;
  const slider = document.getElementById('stepSlider');
  slider.max = currentMaze.frames.length - 1;
  slider.value = 0;
  document.getElementById('mazeSelect').value = id;
  document.getElementById('infoOptimal').textContent = currentMaze.optimal_length;
  document.getElementById('infoModel').textContent = currentMaze.model_length;
  const result = document.getElementById('infoResult');
  if (currentMaze.solved) {
    result.textContent = 'SOLVED';
    result.className = 'info-value solved';
  } else {
    result.textContent = 'FAILED';
    result.className = 'info-value failed';
  }
  render();
}

function randomMaze() {
  const keys = Object.keys(episodes);
  loadMaze(keys[Math.floor(Math.random() * keys.length)]);
}

function setStep(s) {
  currentStep = parseInt(s);
  render();
}

function nextStep() {
  if (!currentMaze) return;
  currentStep = Math.min(currentStep + 1, currentMaze.frames.length - 1);
  document.getElementById('stepSlider').value = currentStep;
  render();
}

function prevStep() {
  currentStep = Math.max(currentStep - 1, 0);
  document.getElementById('stepSlider').value = currentStep;
  render();
}

function togglePlay() {
  playing = !playing;
  document.getElementById('btnPlay').textContent = playing ? 'Pause' : 'Play';
  document.getElementById('btnPlay').classList.toggle('active', playing);
  if (playing) {
    playTimer = setInterval(() => {
      if (currentStep >= currentMaze.frames.length - 1) {
        togglePlay();
        return;
      }
      nextStep();
    }, 300);
  } else {
    clearInterval(playTimer);
  }
}

function render() {
  if (!currentMaze) return;
  const frame = currentMaze.frames[currentStep];
  document.getElementById('stepLabel').textContent =
    `Step ${currentStep}/${currentMaze.frames.length - 1}`;

  // Grid
  drawGrid(frame.grid, currentMaze.optimal_path, currentStep);

  // Info
  document.getElementById('infoAction').textContent = frame.action_name;
  document.getElementById('infoKey').textContent = frame.has_key ? 'Yes' : 'No';
  document.getElementById('infoDoor').textContent = frame.door_open ? 'Open' : 'Closed';

  // Action bars
  const barsDiv = document.getElementById('actionBars');
  barsDiv.innerHTML = '';
  frame.action_probs.forEach((p, i) => {
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = (p * 60) + 'px';
    if (i === frame.action) bar.style.background = '#4ade80';
    barsDiv.appendChild(bar);
  });

  // Phase
  const maxPhase = frame.phase_probs.indexOf(Math.max(...frame.phase_probs));
  for (let i = 0; i < 3; i++) {
    document.getElementById('phase' + i).className = 'phase' + (i === maxPhase ? ' active' : '');
  }

  // Fields
  if (frame.fields) {
    drawField('field2', frame.fields.u2);
    drawField('field1', frame.fields.u1);
    drawField('field0', frame.fields.u0);
  }

  // Trajectory comparison
  drawTrajectory();
}

function drawGrid(grid, optimalPath, step) {
  const canvas = document.getElementById('gridCanvas');
  const ctx = canvas.getContext('2d');
  const size = grid.length;
  const cellSize = canvas.width / size;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw cells
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      ctx.fillStyle = CELL_COLORS[grid[r][c]] || '#1a1f2e';
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      ctx.strokeStyle = '#0f1419';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);
    }
  }

  // Draw optimal path (translucent)
  if (optimalPath) {
    ctx.strokeStyle = 'rgba(74, 222, 128, 0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    optimalPath.forEach((pos, i) => {
      const x = pos[1] * cellSize + cellSize / 2;
      const y = pos[0] * cellSize + cellSize / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Draw model path so far
  const modelPath = currentMaze.frames.slice(0, step + 1).map(f => f.agent_pos);
  if (modelPath.length > 1) {
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    modelPath.forEach((pos, i) => {
      const x = pos[1] * cellSize + cellSize / 2;
      const y = pos[0] * cellSize + cellSize / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Legend
  ctx.font = '10px monospace';
  ctx.fillStyle = '#8892a0';
  ctx.fillText('Green line: optimal | Red line: model', 4, canvas.height - 4);
}

function drawField(canvasId, fieldData) {
  if (!fieldData) return;
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const h = fieldData[0].length;
  const w = fieldData[0][0].length;
  const cellW = canvas.width / w;
  const cellH = canvas.height / h;

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const red = Math.floor(fieldData[0][r][c] * 255);
      const green = Math.floor(fieldData[1][r][c] * 255);
      const blue = Math.floor(fieldData[2][r][c] * 255);
      ctx.fillStyle = `rgb(${red},${green},${blue})`;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }
  }
}

function drawTrajectory() {
  const canvas = document.getElementById('trajCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!currentMaze) return;

  const optLen = currentMaze.optimal_length;
  const modLen = currentMaze.model_length;
  const maxLen = Math.max(optLen, modLen, 1);
  const barW = canvas.width / 2 - 20;

  // Optimal bar
  ctx.fillStyle = '#2a3545';
  ctx.fillRect(10, 30, barW, 30);
  ctx.fillStyle = '#4ade80';
  ctx.fillRect(10, 30, barW * (optLen / maxLen), 30);
  ctx.font = '11px monospace';
  ctx.fillStyle = '#e0e0e0';
  ctx.fillText(`Optimal: ${optLen} steps`, 10, 24);

  // Model bar
  ctx.fillStyle = '#2a3545';
  ctx.fillRect(canvas.width / 2 + 10, 30, barW, 30);
  ctx.fillStyle = currentMaze.solved ? '#c9956b' : '#f87171';
  ctx.fillRect(canvas.width / 2 + 10, 30, barW * (modLen / maxLen), 30);
  ctx.fillText(`Model: ${modLen} steps`, canvas.width / 2 + 10, 24);

  // Current step indicator
  const progress = currentStep / Math.max(modLen - 1, 1);
  ctx.fillStyle = '#ffffff';
  const xPos = canvas.width / 2 + 10 + barW * progress;
  ctx.fillRect(xPos - 1, 28, 2, 34);

  // Efficiency ratio
  const ratio = optLen / Math.max(modLen, 1);
  ctx.fillStyle = '#8892a0';
  ctx.font = '12px monospace';
  ctx.fillText(`Efficiency: ${(ratio * 100).toFixed(0)}% (optimal/model)`, 10, 90);

  // Phase timeline
  ctx.fillStyle = '#8892a0';
  ctx.font = '10px monospace';
  ctx.fillText('Phase timeline:', 10, 120);
  const timelineY = 130;
  const phaseColors = ['#fbbf24', '#a855f7', '#4ade80'];
  const phaseNames = ['Key', 'Door', 'Goal'];
  currentMaze.frames.forEach((f, i) => {
    const maxP = f.phase_probs.indexOf(Math.max(...f.phase_probs));
    const x = 10 + (canvas.width - 20) * (i / Math.max(modLen - 1, 1));
    ctx.fillStyle = phaseColors[maxP];
    ctx.fillRect(x, timelineY, Math.max((canvas.width - 20) / modLen, 2), 12);
  });
  // Legend
  phaseNames.forEach((name, i) => {
    ctx.fillStyle = phaseColors[i];
    ctx.fillRect(10 + i * 80, 155, 10, 10);
    ctx.fillStyle = '#8892a0';
    ctx.fillText(name, 24 + i * 80, 164);
  });
}

init();
</script>
</body>
</html>"""


def run_server(episodes_data, port=8080):
    """Serve the visualization dashboard."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(HTML_TEMPLATE.encode())
            elif self.path == "/api/episodes":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(episodes_data).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # suppress logs

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Dashboard: http://localhost:{port}")
        httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_fluid.pt")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = FluidPlanner(
        d_model=model_args.get("d_model", 64),
        max_steps_per_level=model_args.get("pde_steps", 6),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Solve rate at save: {ckpt.get('solve_rate', 'N/A')}")

    # Generate episodes
    env = MiniGridEnv(size=args.grid_size)
    episodes = {}
    for i in range(args.n_episodes):
        maze_id = 50000 + i
        ep = generate_episode_data(model, env, maze_id, args.device)
        episodes[str(maze_id)] = ep
        status = "SOLVED" if ep["solved"] else "FAILED"
        print(f"  Maze {maze_id}: {status} in {ep['model_length']} steps (optimal: {ep['optimal_length']})")

    solved = sum(1 for e in episodes.values() if e["solved"])
    print(f"\nSolve rate: {solved}/{len(episodes)} = {solved/len(episodes):.0%}")

    # Serve
    run_server(episodes, port=args.port)


if __name__ == "__main__":
    main()
