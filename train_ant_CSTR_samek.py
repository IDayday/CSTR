import os
import torch
import numpy as np
import random
import argparse
import shutil
import json

import time
import gym
from multiworld.envs.mujoco import register_custom_envs as register_mujoco_envs

from utils.logger import Logger
from RIS_TEA_samek import RISTEA
from HER import HERReplayBuffer, PathBuilder
from torch.utils.tensorboard import SummaryWriter


def _as_float_dist(status, key: str = "xy-distance") -> float:
    """Robustly extract a scalar distance from env info dict."""
    if status is None:
        return float("nan")
    v = status.get(key, None)
    if v is None and key == "xy-distance":
        v = status.get("xy_distance", None)
    if v is None:
        return float("nan")
    try:
        arr = np.asarray(v).reshape(-1)
        return float(arr[0])
    except Exception:
        return float(v)


def evalPolicy(
    policy,
    env,
    N: int = 100,
    Tmax: int = 100,
    distance_threshold: float = 0.5,
    logger=None,
    writer=None,
    test_time: int = 0,
    task_type: str = "random",
):
    final_distance = []
    successes = []

    for _ in range(N):
        obs = env.reset()
        state = obs["observation"]
        goal = obs["desired_goal"]
        t = 0

        while True:
            action = policy.select_action(state, goal)
            next_obs, _, _, status = env.step(action)
            state = next_obs["observation"]

            d = _as_float_dist(status)
            done = (d < distance_threshold) or (t >= Tmax)
            t += 1

            if done:
                final_distance.append(d)
                successes.append(1.0 * (d < distance_threshold))
                break

    eval_distance = float(np.mean(final_distance))
    success_rate = float(np.mean(successes))

    if logger is not None:
        logger.store(eval_distance=eval_distance, success_rate=success_rate)

    if writer is not None:
        if task_type == "random":
            writer.add_scalar("random_task_success_rate", success_rate, test_time)
            writer.add_scalar("random_task_eval_distance", eval_distance, test_time)
        elif task_type == "farthest":
            writer.add_scalar("farthest_task_success_rate", success_rate, test_time)
            writer.add_scalar("farthest_task_eval_distance", eval_distance, test_time)

    return eval_distance, success_rate


def sample_and_preprocess_batch(
    replay_buffer,
    batch_size: int = 1024,
    distance_threshold: float = 0.5,
    device: str = "cuda",
):
    batch = replay_buffer.random_batch(batch_size)
    state_batch = batch["observations"]
    action_batch = batch["actions"]
    next_state_batch = batch["next_observations"]
    goal_batch = batch["resampled_goals"]
    reward_batch = batch["rewards"]
    done_batch = batch["terminals"]

    # Sparse rewards: -1 everywhere until reach goal
    reward_shaped = -np.sqrt(np.power(np.array(next_state_batch - goal_batch)[:, :2], 2).sum(-1, keepdims=True))
    done_batch = 1.0 * (reward_shaped > -distance_threshold)
    reward_batch = -np.ones_like(done_batch)

    state_batch = torch.FloatTensor(state_batch).to(device)
    action_batch = torch.FloatTensor(action_batch).to(device)
    reward_batch = torch.FloatTensor(reward_batch).to(device)
    next_state_batch = torch.FloatTensor(next_state_batch).to(device)
    done_batch = torch.FloatTensor(done_batch).to(device)
    goal_batch = torch.FloatTensor(goal_batch).to(device)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="AntU")
    parser.add_argument("--distance_threshold", default=0.5, type=float)
    parser.add_argument("--start_timesteps", default=1e4, type=int)
    parser.add_argument("--eval_freq", default=1e3, type=int)
    parser.add_argument("--max_timesteps", default=5e5, type=int)
    parser.add_argument("--max_episode_length", default=600, type=int)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--replay_buffer_size", default=1e5, type=int)
    parser.add_argument("--n_eval", default=10, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--exp_name", default="CSTR_ant")
    parser.add_argument(
        "--result_group",
        default="CSTR_v1",
        type=str,
        help="Folder name under results/<train_env_name>/ used for TensorBoard, checkpoints, and code snapshots. Do not use RIS for new CSTR runs.",
    )
    parser.add_argument(
        "--result_root",
        default="",
        type=str,
        help="Optional absolute/relative root. If empty, use results/<train_env_name>/<result_group>.",
    )

    # ===== Goal resampling / relabeling ratios (HER replay buffer) =====
    parser.add_argument(
        "--frac_rollout_goals",
        default=0.2,
        type=float,
        help="Fraction of goals in a training batch that are rollout goals (no relabel).",
    )
    parser.add_argument(
        "--frac_env_goals",
        default=0.0,
        type=float,
        help="Fraction of *resampled* goals drawn from env.sample_goals().",
    )
    parser.add_argument(
        "--frac_rb_goals",
        default=0.5,
        type=float,
        help="Fraction of *resampled* goals drawn from replay buffer (random goals).",
    )

    # RIS / TEA shared
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--Lambda", default=0.1, type=float)
    parser.add_argument("--n_ensemble", default=10, type=int, help="Number of random subgoal components M used to build the KL covering prior.")
    parser.add_argument("--critic_n_q", default=2, type=int, help="Number of critic ensemble heads (n_Q).")

    # ---- n_Q>2 ensemble details (SPReD family) ----
    parser.add_argument(
        "--q_target_pair_mode",
        default="fixed2",
        type=str,
        choices=["fixed2", "random2"],
        help="When n_Q>2, choose which 2 heads are used for TD targets: fixed2 (0,1) or random2 per update.",
    )
    parser.add_argument(
        "--q_actor_agg",
        default="auto",
        type=str,
        choices=["auto", "min2", "mean"],
        help="How to aggregate Q_pi in actor loss when n_Q>2. auto->mean, min2->min-of-2, mean->mean-of-ensemble.",
    )

    parser.add_argument("--h_lr", default=1e-4, type=float)
    parser.add_argument("--q_lr", default=1e-3, type=float)
    parser.add_argument("--pi_lr", default=1e-3, type=float)

    parser.add_argument("--log_loss", dest="log_loss", action="store_true")
    parser.add_argument("--no-log_loss", dest="log_loss", action="store_false")
    parser.set_defaults(log_loss=True)

    # -------- TEA-RIS knobs --------
    parser.add_argument("--tea_enabled", default=0, type=int)
    parser.add_argument(
        "--tea_mode",
        default="gated_kl",
        type=str,
        choices=["gated_kl", "choose", "choose_plus", "ra_cgr", "support_mass", "cstr", "ensqfilter", "spredp", "sprede"],
        help=(
            "Teacher-effect weighting mode: "
            "gated_kl (TEA gate), choose (hard choose between RL/KL), choose_plus (conservative hard routing), "
            "ra_cgr (responsibility-aligned routing on the actual KL prior), "
            "support_mass (requires multiple positive teacher candidates), "
            "cstr (Certified Support-Transfer Routing; top-rho + coverage + optional RA veto), "
            "ensqfilter (ensemble Q-filter), spredp / sprede (SPReD-style weights)."
        ),
    )
    parser.add_argument("--tea_q_mode", default="min", type=str, choices=["min", "lcb"])
    parser.add_argument("--tea_q_beta", default=1.0, type=float)
    parser.add_argument("--tea_margin", default=0.0, type=float)
    parser.add_argument("--tea_temp", default=1.0, type=float)
    parser.add_argument("--tea_hard_gate", action="store_true")
    parser.add_argument("--tea_best_of", default=4, type=int)

    # ---- Q-filtered KL prior (align choose gate and KL target) ----
    parser.add_argument(
        "--tea_prior_mode",
        default="all",
        type=str,
        choices=["all", "q_topm"],
        help="How to build the KL prior rho(a|s,g): all (original RIS uniform mixture) or q_topm (Q-filtered sparse mixture).",
    )
    parser.add_argument(
        "--tea_prior_top_m",
        default=0,
        type=int,
        help="Max number of teacher components used in the KL prior (0 -> use all available after filtering).",
    )
    parser.add_argument(
        "--tea_prior_tau",
        default=0.0,
        type=float,
        help="Keep teacher candidates with Q >= Q_best - tau before applying top-M.",
    )
    parser.add_argument(
        "--tea_prior_q_temp",
        default=1.0,
        type=float,
        help="Softmax temperature for mixing weights over the kept teacher components.",
    )
    parser.add_argument(
        "--tea_prior_use_bestofk",
        default=1,
        type=int,
        help="If 1, build KL prior candidates from the first tea_best_of subgoal samples; if 0, use all n_ensemble subgoal samples.",
    )

    parser.add_argument(
        "--tea_prior_eval_mode",
        default="mean",
        type=str,
        choices=["mean", "sample"],
        help="How to evaluate Q for each candidate component when building the Q-filtered prior: mean action (lower variance) or one stochastic sample.",
    )

    parser.add_argument("--tea_warmup_steps", default=0, type=int)

    # diagnostics
    parser.add_argument("--tea_diag_every", default=200, type=int)
    parser.add_argument("--tea_diag_eps", default=1e-3, type=float)
    parser.add_argument("--tea_diag_ema_beta", default=0.95, type=float)
    parser.add_argument("--tea_eval_corr_window", default=25, type=int)
    # Choose+ (conservative hard routing)
    parser.add_argument("--tea_chooseplus_k_low", default=0.0, type=float)
    parser.add_argument("--tea_chooseplus_k_high", default=1.0, type=float)
    parser.add_argument("--tea_chooseplus_eps", default=1e-8, type=float)
    # Diagnostics: delta calibration + gradient conflict
    parser.add_argument("--tea_diag_bins", default=10, type=int)
    parser.add_argument("--tea_diag_log_hist", default=1, type=int)
    parser.add_argument("--tea_diag_log_grad", default=1, type=int)
    parser.add_argument(
        "--tea_minor_coef", type=float, default=0.0,
        help="In hard-routing modes: keep the disadvantaged term with this coefficient. 0.0 means pure winner-take-all."
    )

    # -------- RA-CGR / Support-Mass CGR knobs --------
    parser.add_argument(
        "--tea_ra_eval_mode",
        default="mean",
        type=str,
        choices=["mean", "sample"],
        help="RA-CGR: evaluate each teacher component by tanh(mean) or by one stochastic sample.",
    )
    parser.add_argument(
        "--tea_sm_min_count",
        default=1,
        type=int,
        help="Support-Mass CGR: minimum number of positive candidates required. If <=0, use ceil(tea_sm_min_frac*K).",
    )
    parser.add_argument(
        "--tea_sm_min_frac",
        default=0.0,
        type=float,
        help="Support-Mass CGR: required positive fraction when tea_sm_min_count <= 0.",
    )
    parser.add_argument(
        "--tea_sm_mean_margin",
        default=0.0,
        type=float,
        help="Support-Mass CGR: required mean positive advantage margin after candidate filtering.",
    )
    parser.add_argument(
        "--tea_sm_coverage_threshold",
        default=0.0,
        type=float,
        help="Support-Mass CGR: optional threshold on 1-(1-p_hat)^M; <=0 disables this condition.",
    )
    parser.add_argument(
        "--tea_sm_eval_mode",
        default="sample",
        type=str,
        choices=["mean", "sample"],
        help="Support-Mass CGR: evaluate teacher candidates by tanh(mean) or one stochastic sample.",
    )

    # -------- CSTR knobs --------
    parser.add_argument("--tea_cstr_frac", default=0.25, type=float, help="CSTR top-rho support fraction; m=ceil(rho*K).")
    parser.add_argument("--tea_cstr_margin", default=0.0, type=float, help="CSTR threshold on top-rho mean advantage.")
    parser.add_argument("--tea_cstr_pos_margin", default=0.0, type=float, help="CSTR positive support threshold for pos_frac and coverage.")
    parser.add_argument("--tea_cstr_coverage", default=0.0, type=float, help="CSTR hard coverage threshold 1-(1-pos_frac)^M; <=0 disables.")
    parser.add_argument("--tea_cstr_require_pos_count", default=1, type=int, help="If 1, require at least ceil(rho*K) positive candidates.")
    parser.add_argument("--tea_cstr_use_ra_veto", default=0, type=int, help="If 1, add responsibility-aligned KL-field veto.")
    parser.add_argument("--tea_cstr_ra_margin", default=0.0, type=float, help="RA veto passes if RA advantage > -margin.")
    parser.add_argument("--tea_cstr_eval_mode", default="sample", type=str, choices=["mean", "sample"], help="CSTR candidate action evaluation mode.")
    parser.add_argument("--tea_cstr_policy_q", default="same", type=str, choices=["same", "ucb", "max", "mean"], help="CSTR Q aggregation for policy action in legacy q-value delta.")
    parser.add_argument("--tea_cstr_mean_delta_min", default=-1e9, type=float, help="CSTR-SQ guard: require mean(delta_all) > this value; very negative disables.")
    parser.add_argument("--tea_cstr_prior_source", default="independent", type=str, choices=["independent", "samek_all", "samek_subset"], help="How to build the KL prior for CSTR: independent fresh sample (paper default), samek_all (reuse all K screening supports), or samek_subset (reuse a random subset from the same K support pool).")
    parser.add_argument("--tea_cstr_prior_subset_m", default=10, type=int, help="When tea_cstr_prior_source=samek_subset, randomly choose this many prior components from the same K support pool.")

    # -------- CSTR local density-field certificate --------
    parser.add_argument("--tea_cstr_use_field_veto", default=0, type=int, help="If 1, add responsibility-weighted local reverse-KL density-field certificate.")
    parser.add_argument("--tea_cstr_field_lambda", default=1.0, type=float, help="lambda in the local field test: Delta_loc + lambda * U_rho > margin.")
    parser.add_argument("--tea_cstr_field_margin", default=0.0, type=float, help="Margin in the local field certificate.")
    parser.add_argument("--tea_cstr_field_mode", default="sum", type=str, choices=["sum", "strict"], help="sum: Delta_loc+lambda*U_rho; strict: Delta_loc only.")
    parser.add_argument("--tea_cstr_field_eval_mode", default="mean", type=str, choices=["mean", "sample"], help="Evaluate prior components by target mean or stochastic sample for local field.")

    # -------- CSTR critic calibration --------
    parser.add_argument("--tea_gate_critic", default="online", type=str, choices=["online", "target"], help="Critic used for CSTR gate/certificate; target is a cheap EMA gate critic.")
    parser.add_argument("--tea_calib_adv_mode", default="q_value", type=str, choices=["q_value", "adv_mean", "adv_lcb", "adv_min"], help="Lower-advantage estimator used by CSTR certificates.")
    parser.add_argument("--tea_calib_beta", default=1.0, type=float, help="Beta for advantage-LCB: mean(delta_heads)-beta*std(delta_heads)-b_t.")
    parser.add_argument("--tea_calib_penalty", default=0.0, type=float, help="Constant calibration penalty b_t subtracted from CSTR lower advantages.")
    parser.add_argument("--tea_calib_td_coef", default=0.0, type=float, help="Coefficient for running TD-residual calibration penalty.")
    parser.add_argument("--tea_calib_td_quantile", default=0.8, type=float, help="Batch TD-residual quantile tracked for calibration.")
    parser.add_argument("--tea_calib_td_ema_beta", default=0.98, type=float, help="EMA beta for TD-residual calibration statistic.")
    parser.add_argument("--tea_calib_td_clip", default=10.0, type=float, help="Clip dynamic TD calibration penalty; <=0 disables clipping.")

    # -------- SPReD baseline fidelity knobs --------
    parser.add_argument(
        "--spred_strict",
        default=1,
        type=int,
        help=(
            "If 1 and tea_mode is spred*, match SPReD-style Q-ensemble details "
            "(mean actor Q, random2 TD-target heads, weights computed with grad through policy Q)."
        ),
    )
    parser.add_argument(
        "--spred_detach_w",
        default=0,
        type=int,
        help="If 1, detach SPReD weights from actor gradients (more stable, but deviates from SPReD source).",
    )
    parser.add_argument("--spred_policy_noise", type=float, default=0.2, help="TD3 target policy smoothing noise (SPReD modes only)")
    parser.add_argument("--spred_noise_clip", type=float, default=0.5, help="TD3 noise clip (SPReD modes only)")
    parser.add_argument("--spred_policy_freq", type=int, default=2, help="Delayed policy update frequency (SPReD modes only)")
    parser.add_argument("--spred_lambda1", type=float, default=1.0, help="SPReD actor RL-term coefficient (lambda1)")
    parser.add_argument("--spred_lambda2", type=float, default=1.0, help="SPReD actor imitation-term coefficient (lambda2)")

    parser.add_argument(
        "--grad_clip_norm",
        default=0.0,
        type=float,
        help="If >0, clip grad norm for actor/critic/subgoal to this value (stability/debug).",
    )

    # -------- Optional oracle (expert) gating args (ablation only) --------
    parser.add_argument("--tea_oracle_path", default="", type=str)
    parser.add_argument("--tea_oracle_use_for_gating", default=0, type=int)
    parser.add_argument("--tea_oracle_margin", default=0.0, type=float)

    # -------- Compact direct JSONL diagnostics --------
    parser.add_argument(
        "--compact_log",
        default=1,
        type=int,
        help="If 1, write compact train/eval JSONL diagnostics directly under each run folder/compact_logs.",
    )
    parser.add_argument(
        "--compact_log_every",
        default=0,
        type=int,
        help="Training updates between compact train diagnostic rows. 0 -> use tea_diag_every.",
    )
    parser.add_argument(
        "--compact_log_flush_every",
        default=1,
        type=int,
        help="Flush compact JSONL files every N written rows.",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    spred_modes = {"ensqfilter", "spredp", "sprede"}

    # Safety: keep baseline/gate/choose identical to original RIS/TEA update (n_Q=2).
    if args.tea_mode not in spred_modes:
        if int(args.critic_n_q) != 2:
            print(
                f"[WARN] tea_mode={args.tea_mode} uses original RIS/TEA update (n_Q=2). "
                f"Forcing critic_n_q from {args.critic_n_q} -> 2."
            )
            args.critic_n_q = 2

    # SPReD family expects an ensemble; n_Q=10 is canonical in the original SPReD code.
    if args.tea_mode in spred_modes and int(args.critic_n_q) == 2:
        print("[WARN] SPReD family typically uses n_Q=10. You are running n_Q=2 (allowed, but may underperform).")

    if args.tea_mode in spred_modes and int(args.critic_n_q) < 2:
        raise ValueError("For SPReD modes, critic_n_q must be >= 2.")

    if args.tea_mode == "ra_cgr" and args.tea_prior_mode != "all":
        raise ValueError("tea_mode=ra_cgr requires --tea_prior_mode all, because RA-CGR gates the exact random covering prior used by the KL estimator.")

    if args.tea_mode == "support_mass":
        if int(args.tea_best_of) < 1:
            raise ValueError("support_mass requires --tea_best_of >= 1.")
        if int(args.tea_sm_min_count) <= 0 and float(args.tea_sm_min_frac) <= 0.0:
            raise ValueError("support_mass requires tea_sm_min_count > 0 or tea_sm_min_frac > 0.")

    if args.tea_mode == "cstr":
        if int(args.tea_best_of) < 1:
            raise ValueError("cstr requires --tea_best_of >= 1.")
        if not (0.0 < float(args.tea_cstr_frac) <= 1.0):
            raise ValueError("cstr requires --tea_cstr_frac in (0, 1].")
        if float(args.tea_cstr_coverage) > 1.0:
            raise ValueError("cstr requires --tea_cstr_coverage <= 1.")
        if bool(args.tea_cstr_use_ra_veto) and args.tea_prior_mode != "all":
            raise ValueError("cstr with RA veto requires --tea_prior_mode all.")
        if bool(args.tea_cstr_use_field_veto) and args.tea_prior_mode != "all":
            raise ValueError("cstr with field veto requires --tea_prior_mode all.")
        if str(args.tea_cstr_prior_source) != "independent" and args.tea_prior_mode != "all":
            raise ValueError("same-K prior reuse requires --tea_prior_mode all.")
        if int(args.tea_cstr_prior_subset_m) < 1:
            raise ValueError("tea_cstr_prior_subset_m must be >= 1.")
        if float(args.tea_calib_td_quantile) < 0.0 or float(args.tea_calib_td_quantile) > 1.0:
            raise ValueError("tea_calib_td_quantile must be in [0,1].")

    # Basic validation for goal resampling ratios
    for name in ["frac_rollout_goals", "frac_env_goals", "frac_rb_goals"]:
        v = float(getattr(args, name))
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be within [0,1], got {v}.")
    if args.frac_env_goals + args.frac_rb_goals > 1.0 + 1e-8:
        raise ValueError(
            "frac_env_goals + frac_rb_goals must be <= 1.0 (fractions of the resampled part). "
            f"Got {args.frac_env_goals + args.frac_rb_goals:.4f}."
        )

    # Select environment
    if args.env_name == "AntU":
        train_env_name = "AntULongTrainEnv-v0"
        test_env_name = "AntULongTrainEnv-v0"
        test2_env_name = "AntULongTestEnv-v0"
    elif args.env_name == "AntU-fixed":
        train_env_name = "AntULongTrainEnv-v1"
        test_env_name = "AntULongTrainEnv-v1"
        test2_env_name = "AntULongTestEnv-v1"
    elif args.env_name == "AntFb":
        train_env_name = "AntFbMedTrainEnv-v1"
        test_env_name = "AntFbMedTrainEnv-v1"
        test2_env_name = "AntFbMedTestEnv-v1"
    elif args.env_name == "AntFb-fixed":
        train_env_name = "AntFbMedTrainEnv-v100"
        test_env_name = "AntFbMedTestEnv-v100"
        test2_env_name = test_env_name
    elif args.env_name == "AntMaze":
        train_env_name = "AntMazeMedTrainEnv-v1"
        test_env_name = "AntMazeMedTrainEnv-v1"
        test2_env_name = "AntMazeMedTestEnv-v1"
    elif args.env_name == "AntMaze-fixed":
        train_env_name = "AntMazeMedTrainEnv-v100"
        test_env_name = "AntMazeMedTestEnv-v100"
        test2_env_name = test_env_name
    elif args.env_name == "AntFg":
        train_env_name = "AntFgMedTrainEnv-v1"
        test_env_name = "AntFgMedTrainEnv-v1"
        test2_env_name = "AntFgMedTestEnv-v1"
    elif args.env_name == "AntFg-fixed":
        train_env_name = "AntFgMedTrainEnv-v100"
        test_env_name = "AntFgMedTestEnv-v100"
        test2_env_name = test_env_name
    elif args.env_name == "AntCr":
        train_env_name = "AntCrossTrainEnv-v0"
        test_env_name = "AntCrossTestEnv-v0"
        test2_env_name = test_env_name
    elif args.env_name == "AntPi":
        train_env_name = "AntPiTrainEnv-v0"
        test_env_name = "AntPiTestEnv-v0"
        test2_env_name = test_env_name
    else:
        raise ValueError(f"Unknown env_name: {args.env_name}")

    print("Environments:", train_env_name, test_env_name, test2_env_name)

    # Seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Initialize environment
    register_mujoco_envs()
    vectorized = True

    env = gym.make(train_env_name)
    test_env = gym.make(test_env_name)
    d_test_env = gym.make(test2_env_name)

    action_dim = env.action_space.shape[0]
    state_dim = 31

    ex_time = time.strftime("%m-%d_%H-%M-%S", time.localtime())
    if args.result_root:
        result_root = args.result_root
    else:
        result_root = os.path.join("results", train_env_name, args.result_group)
    folder = os.path.join(result_root, args.exp_name, ex_time)
    code_dir = os.path.join(folder, "code")
    ckpt_dir = os.path.join(folder, "checkpoints")
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Run folder: {folder}")

    # Save machine-readable run metadata next to TensorBoard events. This makes
    # multi-environment exports and duplicate detection independent of log text.
    run_meta = {
        "args": vars(args),
        "train_env_name": train_env_name,
        "test_env_name": test_env_name,
        "farthest_test_env_name": test2_env_name,
        "run_folder": folder,
        "created_at": ex_time,
    }
    with open(os.path.join(folder, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)

    # Save key .py snapshots for reproducibility inside the same run folder.
    for file in ["RIS_TEA_samek.py", "RIS_TEA.py", "HER.py", "Models.py", "train_ant_CSTR_samek.py", "train_ant_CSTR.py", "train_ant_TEA.py"]:
        if os.path.isfile(file):
            shutil.copy(file, os.path.join(code_dir, os.path.basename(file)))

    # Logger
    logger = Logger(vars(args))
    writer = SummaryWriter(folder)

    policy = RISTEA(
        state_dim=state_dim,
        action_dim=action_dim,
        critic_n_q=args.critic_n_q,
        q_target_pair_mode=args.q_target_pair_mode,
        q_actor_agg=args.q_actor_agg,
        alpha=args.alpha,
        Lambda=args.Lambda,
        n_ensemble=int(args.n_ensemble),
        h_lr=args.h_lr,
        q_lr=args.q_lr,
        pi_lr=args.pi_lr,
        device=args.device,
        logger=logger if args.log_loss else None,
        tea_enabled=bool(args.tea_enabled),
        tea_mode=args.tea_mode,
        tea_q_mode=args.tea_q_mode,
        tea_q_beta=args.tea_q_beta,
        tea_margin=args.tea_margin,
        tea_temp=args.tea_temp,
        tea_hard_gate=args.tea_hard_gate,
        tea_best_of=args.tea_best_of,
        tea_prior_mode=args.tea_prior_mode,
        tea_prior_top_m=args.tea_prior_top_m,
        tea_prior_tau=float(args.tea_prior_tau),
        tea_prior_q_temp=float(args.tea_prior_q_temp),
        tea_prior_use_bestofk=bool(args.tea_prior_use_bestofk),
        tea_prior_eval_mode=str(args.tea_prior_eval_mode),
        tea_warmup_steps=args.tea_warmup_steps,
        tea_diag_every=args.tea_diag_every,
        tea_diag_eps=args.tea_diag_eps,
        tea_diag_ema_beta=args.tea_diag_ema_beta,
        tea_eval_corr_window=args.tea_eval_corr_window,
        tea_chooseplus_k_low=float(args.tea_chooseplus_k_low),
        tea_chooseplus_k_high=float(args.tea_chooseplus_k_high),
        tea_chooseplus_eps=float(args.tea_chooseplus_eps),
        tea_diag_bins=int(args.tea_diag_bins),
        tea_diag_log_hist=bool(args.tea_diag_log_hist),
        tea_diag_log_grad=bool(args.tea_diag_log_grad),
        tea_oracle_path=(args.tea_oracle_path if args.tea_oracle_path != "" else None),
        tea_oracle_use_for_gating=bool(args.tea_oracle_use_for_gating),
        tea_oracle_margin=args.tea_oracle_margin,
        tea_minor_coef=args.tea_minor_coef,
        tea_ra_eval_mode=str(args.tea_ra_eval_mode),
        tea_sm_min_count=int(args.tea_sm_min_count),
        tea_sm_min_frac=float(args.tea_sm_min_frac),
        tea_sm_mean_margin=float(args.tea_sm_mean_margin),
        tea_sm_coverage_threshold=float(args.tea_sm_coverage_threshold),
        tea_sm_eval_mode=str(args.tea_sm_eval_mode),
        tea_cstr_frac=float(args.tea_cstr_frac),
        tea_cstr_margin=float(args.tea_cstr_margin),
        tea_cstr_pos_margin=float(args.tea_cstr_pos_margin),
        tea_cstr_coverage=float(args.tea_cstr_coverage),
        tea_cstr_require_pos_count=bool(args.tea_cstr_require_pos_count),
        tea_cstr_use_ra_veto=bool(args.tea_cstr_use_ra_veto),
        tea_cstr_ra_margin=float(args.tea_cstr_ra_margin),
        tea_cstr_eval_mode=str(args.tea_cstr_eval_mode),
        tea_cstr_policy_q=str(args.tea_cstr_policy_q),
        tea_cstr_mean_delta_min=float(args.tea_cstr_mean_delta_min),
        tea_cstr_prior_source=str(args.tea_cstr_prior_source),
        tea_cstr_prior_subset_m=int(args.tea_cstr_prior_subset_m),
        tea_cstr_use_field_veto=bool(args.tea_cstr_use_field_veto),
        tea_cstr_field_lambda=float(args.tea_cstr_field_lambda),
        tea_cstr_field_margin=float(args.tea_cstr_field_margin),
        tea_cstr_field_mode=str(args.tea_cstr_field_mode),
        tea_cstr_field_eval_mode=str(args.tea_cstr_field_eval_mode),
        tea_gate_critic=str(args.tea_gate_critic),
        tea_calib_adv_mode=str(args.tea_calib_adv_mode),
        tea_calib_beta=float(args.tea_calib_beta),
        tea_calib_penalty=float(args.tea_calib_penalty),
        tea_calib_td_coef=float(args.tea_calib_td_coef),
        tea_calib_td_quantile=float(args.tea_calib_td_quantile),
        tea_calib_td_ema_beta=float(args.tea_calib_td_ema_beta),
        tea_calib_td_clip=float(args.tea_calib_td_clip),
        spred_strict=bool(args.spred_strict),
        spred_detach_w=bool(args.spred_detach_w),
        spred_policy_noise=float(args.spred_policy_noise),
        spred_noise_clip=float(args.spred_noise_clip),
        spred_policy_freq=int(args.spred_policy_freq),
        spred_lambda1=float(args.spred_lambda1),
        spred_lambda2=float(args.spred_lambda2),
        grad_clip_norm=float(args.grad_clip_norm),
        writer=writer,
    )

    if bool(args.compact_log):
        try:
            policy.set_compact_logger(
                folder,
                run_meta=run_meta,
                log_every=(int(args.compact_log_every) if int(args.compact_log_every) > 0 else int(args.tea_diag_every)),
                flush_every=int(args.compact_log_flush_every),
            )
            print(f"Compact diagnostics: {os.path.join(folder, 'compact_logs')}")
        except Exception as exc:
            print(f"[WARN] compact diagnostics disabled: {exc}")

    # Replay buffer and path builder
    replay_buffer = HERReplayBuffer(
        max_size=args.replay_buffer_size,
        env=env,
        fraction_goals_are_rollout_goals=args.frac_rollout_goals,
        fraction_resampled_goals_are_env_goals=args.frac_env_goals,
        fraction_resampled_goals_are_replay_buffer_goals=args.frac_rb_goals,
        ob_keys_to_save=["state_achieved_goal", "state_desired_goal"],
        desired_goal_keys=["desired_goal", "state_desired_goal"],
        observation_key="observation",
        desired_goal_key="desired_goal",
        achieved_goal_key="achieved_goal",
        vectorized=vectorized,
    )
    path_builder = PathBuilder()

    # Reset env
    obs = env.reset()
    state = obs["observation"]
    goal = obs["desired_goal"]
    episode_timesteps = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state, goal)

        # Step
        next_obs, reward, _, status = env.step(action)
        next_state = next_obs["observation"]
        d = _as_float_dist(status)
        done = d < args.distance_threshold

        path_builder.add_all(
            observations=obs,
            actions=action,
            rewards=reward,
            next_observations=next_obs,
            terminals=[1.0 * done],
        )

        state = next_state
        obs = next_obs

        # Train
        if t >= args.batch_size and t >= args.start_timesteps:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch = sample_and_preprocess_batch(
                replay_buffer,
                batch_size=args.batch_size,
                distance_threshold=args.distance_threshold,
                device=args.device,
            )
            subgoal_batch = torch.FloatTensor(replay_buffer.random_state_batch(args.batch_size)).to(args.device)
            policy.train(state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch, subgoal_batch, t)

        if done or episode_timesteps >= args.max_episode_length:
            replay_buffer.add_path(path_builder.get_all_stacked())
            path_builder = PathBuilder()
            logger.store(t=t, distance=d)

            obs = env.reset()
            state = obs["observation"]
            goal = obs["desired_goal"]
            episode_timesteps = 0

        if (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            # random tasks
            eval_distance, success_rate = evalPolicy(
                policy,
                test_env,
                N=args.n_eval * 10,
                Tmax=args.max_episode_length,
                distance_threshold=args.distance_threshold,
                logger=logger,
                writer=writer,
                test_time=t,
                task_type="random",
            )
            try:
                policy.record_eval(success_rate, t=int(t), tag="random", eval_distance=eval_distance)
            except Exception:
                pass

            # hardest tasks
            eval_distance, success_rate = evalPolicy(
                policy,
                d_test_env,
                N=args.n_eval,
                Tmax=args.max_episode_length,
                distance_threshold=args.distance_threshold,
                logger=logger,
                writer=writer,
                test_time=t,
                task_type="farthest",
            )
            try:
                policy.record_eval(success_rate, t=int(t), tag="farthest", eval_distance=eval_distance)
            except Exception:
                pass

            print(f"CSTR | {logger}")

            # Save everything in this run folder, not under RIS/RISTEA.
            logger.save(os.path.join(folder, "log.pkl"))
            policy.save(ckpt_dir + os.sep)


if __name__ == "__main__":
    main()