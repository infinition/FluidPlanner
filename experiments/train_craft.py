"""
Training script for CraftWorld (variable-depth hierarchical planning).

This is the "incontestable" experiment: recipe depth varies from 1 to 3,
so the model must adapt its hierarchical decomposition per episode.
No phase labels are provided -- emergence only.

Usage:
  python experiments/train_craft.py --model fluid --epochs 200 --grid-size 20
  python experiments/train_craft.py --model flat --epochs 200 --grid-size 20
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fluidplanner.envs.craftworld import CraftWorldEnv, generate_craft_dataset, RECIPE_TREE
from fluidplanner.planner import FluidPlanner, FlatBaseline


def collate(dataset, device):
    states, actions, depths = [], [], []
    for traj in dataset:
        for s, a, d in traj:
            states.append(s)
            actions.append(a)
            depths.append(d)
    return (
        torch.stack(states).to(device),
        torch.tensor(actions, dtype=torch.long, device=device),
        torch.tensor(depths, dtype=torch.long, device=device),
    )


def evaluate_episodes(model, n_episodes=30, grid_size=20, device="cpu"):
    """Run full episodes at each difficulty and measure solve rate."""
    env = CraftWorldEnv(size=grid_size)
    results = {1: {"solved": 0, "total": 0}, 2: {"solved": 0, "total": 0}, 3: {"solved": 0, "total": 0}}

    model.eval()
    for i in range(n_episodes):
        for difficulty in [1, 2, 3]:
            env.reset(difficulty=difficulty, seed=80000 + i * 10 + difficulty)
            optimal = env.optimal_steps()
            if len(optimal) < 2:
                continue

            env.reset(difficulty=difficulty, seed=80000 + i * 10 + difficulty)
            results[difficulty]["total"] += 1
            max_steps = max(len(optimal) * 3, 50)

            for step in range(max_steps):
                state = env.to_tensor(device).unsqueeze(0)
                with torch.no_grad():
                    out = model(state)
                    action = out["action_logits"].argmax(dim=-1).item()
                _, _, done = env.step(action)
                if done:
                    if env.delivered:
                        results[difficulty]["solved"] += 1
                    break

    model.train()
    return results


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate datasets
    print(f"Generating training data (grid={args.grid_size})...")
    train_data = generate_craft_dataset(args.n_train, args.grid_size, seed=42)
    val_data = generate_craft_dataset(args.n_val, args.grid_size, seed=9999)

    train_s, train_a, train_d = collate(train_data, device)
    val_s, val_a, val_d = collate(val_data, device)
    print(f"  Train: {len(train_s)} steps | Val: {len(val_s)} steps")

    # Model (12 input channels for CraftWorld)
    if args.model == "fluid":
        model = FluidPlanner(
            grid_channels=12,
            d_model=args.d_model,
            max_steps_per_level=args.pde_steps,
            n_actions=6,  # up, down, left, right, craft, noop
        ).to(device)
    else:
        model = FlatBaseline(
            grid_channels=12,
            d_model=args.d_model,
            n_actions=6,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Params: {n_params:,}")
    print(f"  NO phase supervision (emergence test)")

    # Freeze phase head
    if hasattr(model, "phase_head"):
        for p in model.phase_head.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"craft_{args.model}_d{args.d_model}_g{args.grid_size}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    best_total_solve = 0.0
    batch_size = args.batch_size
    n_train = len(train_s)

    print(f"\nTraining for {args.epochs} epochs")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_s = train_s[perm[i:i+batch_size]]
            batch_a = train_a[perm[i:i+batch_size]]

            out = model(batch_s)
            loss = F.cross_entropy(out["action_logits"], batch_a)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        # Val accuracy
        model.eval()
        with torch.no_grad():
            val_out = model(val_s)
            val_acc = (val_out["action_logits"].argmax(-1) == val_a).float().mean().item()
        model.train()

        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Val/Action_Acc", val_acc, epoch)

        # Full episode eval
        if epoch % args.eval_interval == 0 or epoch == 1:
            results = evaluate_episodes(model, n_episodes=args.eval_episodes,
                                        grid_size=args.grid_size, device=device)

            total_solved = sum(r["solved"] for r in results.values())
            total_episodes = sum(r["total"] for r in results.values())
            total_rate = total_solved / max(total_episodes, 1)

            for d in [1, 2, 3]:
                rate = results[d]["solved"] / max(results[d]["total"], 1)
                writer.add_scalar(f"Eval/Solve_Depth{d}", rate, epoch)

            writer.add_scalar("Eval/Solve_Total", total_rate, epoch)

            if total_rate > best_total_solve:
                best_total_solve = total_rate
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "solve_rates": {d: results[d]["solved"] / max(results[d]["total"], 1) for d in [1, 2, 3]},
                    "total_solve_rate": total_rate,
                    "args": vars(args),
                }, f"checkpoints/best_{args.model}_craft.pt")

            d1 = results[1]["solved"] / max(results[1]["total"], 1)
            d2 = results[2]["solved"] / max(results[2]["total"], 1)
            d3 = results[3]["solved"] / max(results[3]["total"], 1)

            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  acc={val_acc:.3f}  "
                  f"depth1={d1:.0%}  depth2={d2:.0%}  depth3={d3:.0%}  "
                  f"total={total_rate:.0%}  best={best_total_solve:.0%}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  acc={val_acc:.3f}")

    writer.close()
    print(f"\nDone. Best total solve: {best_total_solve:.0%}")
    print(f"TensorBoard: tensorboard --logdir=runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fluid", "flat"], default="fluid")
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--pde-steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()
    train(args)
