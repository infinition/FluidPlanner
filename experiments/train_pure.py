"""
Pure Emergence Experiment.

ZERO hierarchical bias. Single PDE, uniform channels.
After training, analyze whether temporal stratification emerged:
- Do some channels become "slow" (plan-like)?
- Do others become "fast" (action-like)?

If yes: genuine emergence from reaction-diffusion dynamics.
If no: hierarchy needs architectural support.

Either result is scientifically valuable.

Usage:
  python experiments/train_pure.py --epochs 100 --grid-size 16
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fluidplanner.envs.minigrid import MiniGridEnv, generate_dataset
from fluidplanner.planner_pure import FluidPlannerPure


def collate(dataset, device):
    states, actions, phases = [], [], []
    for traj in dataset:
        for s, a, p in traj:
            states.append(s)
            actions.append(a)
            phases.append(p)
    return (
        torch.stack(states).to(device),
        torch.tensor(actions, dtype=torch.long, device=device),
        torch.tensor(phases, dtype=torch.long, device=device),
    )


def evaluate_episodes(model, n_episodes=50, grid_size=16, device="cpu"):
    env = MiniGridEnv(size=grid_size)
    solved = 0
    for i in range(n_episodes):
        env.reset(maze_id=99999 + i)
        optimal = env.optimal_trajectory()
        if len(optimal) < 3:
            continue
        env.reset(maze_id=99999 + i)
        for step in range(len(optimal) * 3):
            state = env.to_tensor(device).unsqueeze(0)
            with torch.no_grad():
                out = model(state)
                action = out["action_logits"].argmax(-1).item()
            _, _, done = env.step(action)
            if done:
                solved += 1
                break
    return solved / n_episodes


def analyze_emergence(model, val_states, writer, epoch):
    """
    THE key analysis: measure per-channel temporal dynamics.
    If channels self-organize into fast/slow groups, hierarchy emerged.
    """
    model.eval()
    with torch.no_grad():
        # Run on a batch to get channel velocities
        batch = val_states[:64]
        out = model(batch)
        velocity = out["channel_velocity"]  # (C,)

        # Sort channels by velocity
        sorted_vel, sorted_idx = velocity.sort()

        # Compute stratification metric:
        # ratio of fastest 25% to slowest 25%
        n = len(velocity)
        q = n // 4
        slow_mean = sorted_vel[:q].mean().item()
        fast_mean = sorted_vel[-q:].mean().item()
        stratification = fast_mean / (slow_mean + 1e-8)

        # Log
        writer.add_scalar("Emergence/Stratification_Ratio", stratification, epoch)
        writer.add_scalar("Emergence/Slowest_25pct", slow_mean, epoch)
        writer.add_scalar("Emergence/Fastest_25pct", fast_mean, epoch)
        writer.add_scalar("Emergence/Velocity_Std", velocity.std().item(), epoch)
        writer.add_scalar("Emergence/Velocity_Mean", velocity.mean().item(), epoch)

        # Log histogram of channel velocities
        writer.add_histogram("Emergence/Channel_Velocities", velocity, epoch)

        # Log diffusion coefficients -- did they diverge from uniform init?
        diff_coeffs = 0.25 * torch.sigmoid(model.pde.diffusion.diff_logits)  # (n_dilations, C)
        for dil_idx in range(diff_coeffs.shape[0]):
            dil_coeffs = diff_coeffs[dil_idx]
            writer.add_scalar(f"Diffusion/Dilation{dil_idx}_Std", dil_coeffs.std().item(), epoch)
            writer.add_scalar(f"Diffusion/Dilation{dil_idx}_Min", dil_coeffs.min().item(), epoch)
            writer.add_scalar(f"Diffusion/Dilation{dil_idx}_Max", dil_coeffs.max().item(), epoch)
            writer.add_histogram(f"Diffusion/Dilation{dil_idx}_Coeffs", dil_coeffs, epoch)

    model.train()
    return stratification, slow_mean, fast_mean


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Generating {args.n_train} training mazes...")
    train_data = generate_dataset(args.n_train, args.grid_size, seed=42)
    val_data = generate_dataset(args.n_val, args.grid_size, seed=9999)

    train_s, train_a, train_p = collate(train_data, device)
    val_s, val_a, val_p = collate(val_data, device)
    print(f"  Train: {len(train_s)} steps | Val: {len(val_s)} steps")

    model = FluidPlannerPure(
        grid_channels=6,
        d_model=args.d_model,
        max_steps=args.pde_steps,
        n_actions=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: PureEmergence | Params: {n_params:,} | d_model: {args.d_model}")
    print(f"  PDE steps: {args.pde_steps} | All channels UNIFORM at init")
    print(f"  ZERO hierarchical bias")

    # Verify uniform initialization
    diff_coeffs = 0.25 * torch.sigmoid(model.pde.diffusion.diff_logits)
    print(f"  Initial D range: [{diff_coeffs.min():.4f}, {diff_coeffs.max():.4f}] (should be ~0.05)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir=f"runs/pure_d{args.d_model}_s{args.pde_steps}")

    best_solve = 0.0
    n_train = len(train_s)

    print(f"\nTraining for {args.epochs} epochs")
    print("=" * 70)
    print("Watching for emergence: Stratification_Ratio > 3.0 = strong signal")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, args.batch_size):
            batch_s = train_s[perm[i:i+args.batch_size]]
            batch_a = train_a[perm[i:i+args.batch_size]]

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

        if epoch % args.eval_interval == 0 or epoch == 1:
            # Emergence analysis
            strat, slow, fast = analyze_emergence(model, val_s, writer, epoch)

            # Solve rate
            solve = evaluate_episodes(model, n_episodes=args.eval_episodes,
                                      grid_size=args.grid_size, device=device)
            writer.add_scalar("Eval/Solve_Rate", solve, epoch)

            if solve > best_solve:
                best_solve = solve
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "solve_rate": solve,
                    "stratification": strat,
                    "args": vars(args),
                }, "checkpoints/best_pure.pt")

            # Emergence verdict
            if strat > 5.0:
                verdict = "STRONG EMERGENCE"
            elif strat > 3.0:
                verdict = "moderate emergence"
            elif strat > 1.5:
                verdict = "weak signal"
            else:
                verdict = "no emergence"

            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  acc={val_acc:.3f}  "
                  f"solve={solve:.0%}  strat={strat:.1f}x  "
                  f"(slow={slow:.4f} fast={fast:.4f})  [{verdict}]  "
                  f"best={best_solve:.0%}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  acc={val_acc:.3f}")

    # Final emergence report
    print("\n" + "=" * 70)
    print("EMERGENCE REPORT")
    print("=" * 70)
    strat, slow, fast = analyze_emergence(model, val_s, writer, args.epochs + 1)

    velocity = None
    model.eval()
    with torch.no_grad():
        out = model(val_s[:64])
        velocity = out["channel_velocity"]

    sorted_vel, sorted_idx = velocity.sort()
    n = len(velocity)
    q = n // 4

    print(f"  Channels: {n}")
    print(f"  Slowest 25% (channels {sorted_idx[:q].tolist()}):")
    print(f"    Mean velocity: {sorted_vel[:q].mean():.6f}")
    print(f"  Fastest 25% (channels {sorted_idx[-q:].tolist()}):")
    print(f"    Mean velocity: {sorted_vel[-q:].mean():.6f}")
    print(f"  Stratification ratio: {strat:.2f}x")
    print()

    if strat > 5.0:
        print("  RESULT: STRONG TEMPORAL STRATIFICATION EMERGED")
        print("  Some channels spontaneously became slow (plan-like)")
        print("  and others fast (action-like), from uniform initialization.")
        print("  This is genuine emergence of hierarchical temporal structure.")
    elif strat > 3.0:
        print("  RESULT: MODERATE STRATIFICATION")
        print("  Channels show meaningful velocity differences.")
        print("  Partial emergence of temporal hierarchy.")
    elif strat > 1.5:
        print("  RESULT: WEAK SIGNAL")
        print("  Some velocity variation but not clearly stratified.")
    else:
        print("  RESULT: NO EMERGENCE")
        print("  All channels evolve at similar speeds.")
        print("  Hierarchy requires architectural support.")

    print()
    print("  Both results are scientifically valuable:")
    print("  - Emergence = PDE dynamics naturally create temporal hierarchy")
    print("  - No emergence = hierarchy needs explicit architectural bias")
    print("=" * 70)

    # === Channel Ablation Analysis ===
    # Per ChatGPT/Gemini: if cutting slow channels hurts long-term coherence
    # and cutting fast channels hurts immediate action, that proves functional
    # specialization beyond mere temporal stratification.
    print("\n" + "=" * 70)
    print("CHANNEL ABLATION ANALYSIS")
    print("=" * 70)

    slow_idx = sorted_idx[:q].tolist()
    fast_idx = sorted_idx[-q:].tolist()
    mid_idx = sorted_idx[q:-q].tolist()

    # Full model baseline
    solve_full = evaluate_episodes(model, n_episodes=args.eval_episodes,
                                   grid_size=args.grid_size, device=device)
    print(f"  Full model solve rate: {solve_full:.0%}")

    # Ablation helper: zero out channel groups
    def ablate_and_eval(channel_indices, label):
        """Zero out specific channels in the encoder output and evaluate."""
        hooks = []
        def hook_fn(module, input, output):
            output = output.clone()
            output[:, channel_indices, :, :] = 0
            return output
        # Hook into encoder output
        h = model.encoder.register_forward_hook(hook_fn)
        hooks.append(h)
        solve = evaluate_episodes(model, n_episodes=args.eval_episodes,
                                  grid_size=args.grid_size, device=device)
        for hk in hooks:
            hk.remove()
        print(f"  Ablate {label} ({len(channel_indices)} ch): solve={solve:.0%}  "
              f"delta={solve - solve_full:+.0%}")
        return solve

    solve_no_slow = ablate_and_eval(slow_idx, "SLOW 25%")
    solve_no_fast = ablate_and_eval(fast_idx, "FAST 25%")
    solve_no_mid = ablate_and_eval(mid_idx, "MIDDLE 50%")

    # Log ablation results
    writer.add_scalar("Ablation/Full", solve_full, args.epochs)
    writer.add_scalar("Ablation/No_Slow_25pct", solve_no_slow, args.epochs)
    writer.add_scalar("Ablation/No_Fast_25pct", solve_no_fast, args.epochs)
    writer.add_scalar("Ablation/No_Mid_50pct", solve_no_mid, args.epochs)

    print()
    if solve_no_fast < solve_no_slow:
        print("  Fast channels are MORE important for action -> functional specialization!")
    elif solve_no_slow < solve_no_fast:
        print("  Slow channels are MORE important -> possible planning/memory role")
    else:
        print("  Equal impact -> no functional differentiation detected")

    # === Velocity Histogram (numerical) ===
    print("\n  Velocity histogram (8 bins):")
    vel_np = velocity.cpu().numpy()
    counts, edges = np.histogram(vel_np, bins=8)
    for i in range(len(counts)):
        bar = "#" * int(counts[i] / max(counts) * 30) if max(counts) > 0 else ""
        print(f"    [{edges[i]:.5f}, {edges[i+1]:.5f}): {counts[i]:3d} {bar}")

    # Save velocity data
    np.savez("checkpoints/emergence_data.npz",
             velocity=vel_np,
             sorted_idx=sorted_idx.cpu().numpy(),
             stratification=strat,
             ablation_full=solve_full,
             ablation_no_slow=solve_no_slow,
             ablation_no_fast=solve_no_fast,
             ablation_no_mid=solve_no_mid)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--pde-steps", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="Random seed (run multiple seeds for reproducibility)")
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)
