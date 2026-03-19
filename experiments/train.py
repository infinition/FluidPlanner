"""
Training script for FluidPlanner.

Imitation learning on optimal A* trajectories in key-door-goal grid worlds.
Logs to TensorBoard for monitoring.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fluidplanner.envs.minigrid import MiniGridEnv, generate_dataset
from fluidplanner.planner import FluidPlanner, FlatBaseline


def collate_trajectories(dataset, device):
    """Flatten trajectories into (state, action, phase) batches."""
    states, actions, phases = [], [], []
    for traj in dataset:
        for state_t, action, phase in traj:
            states.append(state_t)
            actions.append(action)
            phases.append(phase)
    return (
        torch.stack(states).to(device),
        torch.tensor(actions, dtype=torch.long, device=device),
        torch.tensor(phases, dtype=torch.long, device=device),
    )


def evaluate(model, val_states, val_actions, val_phases):
    """Compute action accuracy and phase accuracy."""
    model.eval()
    with torch.no_grad():
        out = model(val_states)
        action_preds = out["action_logits"].argmax(dim=-1)
        action_acc = (action_preds == val_actions).float().mean().item()

        phase_acc = 0.0
        if "phase_logits" in out and out["phase_logits"].shape[-1] == 3:
            phase_preds = out["phase_logits"].argmax(dim=-1)
            phase_acc = (phase_preds == val_phases).float().mean().item()

    model.train()
    return action_acc, phase_acc


def evaluate_full_episodes(model, n_episodes=50, grid_size=16, seed_offset=99999, device="cpu"):
    """Run full episodes and measure solve rate."""
    env = MiniGridEnv(size=grid_size)
    solved = 0
    total_steps = 0

    model.eval()
    for i in range(n_episodes):
        env.reset(maze_id=seed_offset + i)
        optimal = env.optimal_trajectory()
        if len(optimal) < 3:
            continue  # skip unsolvable

        env.reset(maze_id=seed_offset + i)  # re-reset
        max_steps = len(optimal) * 3  # generous budget

        for step in range(max_steps):
            state = env.to_tensor(device).unsqueeze(0)
            with torch.no_grad():
                out = model(state)
                action = out["action_logits"].argmax(dim=-1).item()
            _, _, done = env.step(action)
            if done:
                solved += 1
                total_steps += step + 1
                break

    model.train()
    solve_rate = solved / n_episodes
    avg_steps = total_steps / max(solved, 1)
    return solve_rate, avg_steps


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate datasets
    print(f"Generating {args.n_train} training mazes (grid={args.grid_size})...")
    train_data = generate_dataset(args.n_train, args.grid_size, seed=42)
    val_data = generate_dataset(args.n_val, args.grid_size, seed=9999)
    print(f"  Train: {len(train_data)} trajectories, {sum(len(t) for t in train_data)} steps")
    print(f"  Val:   {len(val_data)} trajectories, {sum(len(t) for t in val_data)} steps")

    train_states, train_actions, train_phases = collate_trajectories(train_data, device)
    val_states, val_actions, val_phases = collate_trajectories(val_data, device)

    # Model
    if args.model == "fluid":
        model = FluidPlanner(
            grid_channels=6,
            d_model=args.d_model,
            max_steps_per_level=args.pde_steps,
            n_actions=4,
        ).to(device)
    else:
        model = FlatBaseline(
            grid_channels=6,
            d_model=args.d_model,
            n_actions=4,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Params: {n_params:,} | d_model: {args.d_model}")
    print(f"  PDE steps/level: {args.pde_steps}")
    print(f"  Mode: NO phase supervision (emergence test)")

    # Freeze phase_head -- no gradient, used as probe only
    if hasattr(model, "phase_head"):
        for p in model.phase_head.parameters():
            p.requires_grad = False

    # Optimizer (phase_head excluded since frozen)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    run_name = f"{args.model}_d{args.d_model}_s{args.pde_steps}_g{args.grid_size}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # Training
    best_solve_rate = 0.0
    batch_size = args.batch_size
    n_train = len(train_states)

    print(f"\nTraining for {args.epochs} epochs, batch_size={batch_size}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_action_loss = 0.0
        epoch_phase_loss = 0.0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(n_train, device=device)
        train_states_shuf = train_states[perm]
        train_actions_shuf = train_actions[perm]
        train_phases_shuf = train_phases[perm]

        for i in range(0, n_train, batch_size):
            batch_s = train_states_shuf[i:i+batch_size]
            batch_a = train_actions_shuf[i:i+batch_size]
            batch_p = train_phases_shuf[i:i+batch_size]

            out = model(batch_s)

            # Action loss ONLY -- no phase supervision
            # The model must discover the hierarchical decomposition on its own
            action_loss = F.cross_entropy(out["action_logits"], batch_a)

            # Phase head exists but receives NO gradient -- used only for probing
            phase_loss = torch.tensor(0.0, device=device)

            loss = action_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_action_loss += action_loss.item()
            epoch_phase_loss += phase_loss.item()
            n_batches += 1

        scheduler.step()

        # Metrics
        avg_loss = epoch_loss / n_batches
        avg_action_loss = epoch_action_loss / n_batches
        avg_phase_loss = epoch_phase_loss / n_batches
        action_acc, phase_acc = evaluate(model, val_states, val_actions, val_phases)

        # TensorBoard
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/Action_Loss", avg_action_loss, epoch)
        writer.add_scalar("Train/Phase_Loss", avg_phase_loss, epoch)
        writer.add_scalar("Val/Action_Acc", action_acc, epoch)
        writer.add_scalar("Val/Phase_Acc", phase_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Full episode evaluation every N epochs
        if epoch % args.eval_interval == 0 or epoch == 1:
            solve_rate, avg_steps = evaluate_full_episodes(
                model, n_episodes=args.eval_episodes,
                grid_size=args.grid_size, device=device,
            )
            writer.add_scalar("Eval/Solve_Rate", solve_rate, epoch)
            writer.add_scalar("Eval/Avg_Steps", avg_steps, epoch)

            if solve_rate > best_solve_rate:
                best_solve_rate = solve_rate
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "solve_rate": solve_rate,
                    "args": vars(args),
                }, f"checkpoints/best_{args.model}.pt")

            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"act_acc={action_acc:.3f}  phase_acc={phase_acc:.3f}  "
                  f"solve={solve_rate:.1%}  steps={avg_steps:.1f}  "
                  f"best={best_solve_rate:.1%}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"act_acc={action_acc:.3f}  phase_acc={phase_acc:.3f}")

    writer.close()
    print(f"\nDone. Best solve rate: {best_solve_rate:.1%}")
    print(f"Checkpoint: checkpoints/best_{args.model}.pt")
    print(f"TensorBoard: tensorboard --logdir=runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FluidPlanner Training")
    parser.add_argument("--model", choices=["fluid", "flat"], default="fluid")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--pde-steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()
    train(args)
