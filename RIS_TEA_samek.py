import numpy as np
import os
import json
import time
import torch
import torch.nn.functional as F
import torch.distributions as D

from Models import GaussianPolicy, EnsembleCritic, LaplacePolicy, Encoder
from utils.data_aug import random_translate


class RISTEA(object):
	"""Teacher-Effect-Aware RIS (TEA-RIS).

	This is a minimal, engineering-friendly extension of the original RIS.

	When `tea_enabled=False`, this class behaves identically to the original RIS.

	When `tea_enabled=True`, we compute a per-sample teacher-effect score
	\\Delta(s,g) using a conservative critic estimate (LCB) comparing:
	- the current policy action a_pi ~ pi(a|s,g)
	- the teacher action a_teach sampled from the subgoal-induced prior (best-of-K)

	Then we gate the KL term:
	- mode='gated_kl': actor loss = alpha * w(\\Delta) * D_KL - Q
	- mode='choose'  : per-sample choose between KL-only and RL-only

	All gates are detached to avoid unstable second-order effects.
	"""

	def __init__(
			self,
			state_dim,
			action_dim,
			critic_n_q=2,
			alpha=0.1,
			Lambda=0.1,
			image_env=False,
			n_ensemble=10,
			gamma=0.99,
			tau=0.005,
			target_update_interval=1,
			h_lr=1e-4,
			q_lr=1e-3,
			pi_lr=1e-4,
			enc_lr=1e-4,
			epsilon=1e-16,
			logger=None,
			device=torch.device("cuda"),
			# ---- TEA-RIS knobs ----
			tea_enabled=False,
			tea_mode="gated_kl",            # ['gated_kl', 'choose', 'choose_plus', 'ra_cgr', 'support_mass', 'cstr', 'ensqfilter', 'spredp', 'sprede']
			tea_q_beta=1.0,                 # LCB: mean - beta * std
			tea_margin=0.0,                 # gate threshold on Delta
			tea_temp=1.0,                   # sigmoid temperature
			tea_hard_gate=False,            # if True, use hard gate I[Delta>margin]
			tea_best_of=4,                  # best-of-K teacher candidates
			# ---- Q-filtered prior for KL (align KL target with Q-gated choose) ----
			tea_prior_mode="all",           # ['all','q_topm']
			tea_prior_top_m=0,               # 0 -> use all available candidates after filtering
			tea_prior_tau=0.0,               # keep candidates with Q >= Q_best - tau
			tea_prior_q_temp=1.0,            # softmax temperature for mixture weights over candidates
			tea_prior_use_bestofk=True,      # if True, build prior from first tea_best_of subgoal samples
			tea_prior_eval_mode="mean",     # ['mean','sample'] how to evaluate Q for candidate components
			tea_warmup_steps=0,             # do not gate before this step
			tea_q_mode="min",              # ['min', 'lcb'] used for Delta computation
			# ---- Ensemble aggregation controls (stability for n_Q>2) ----
			q_target_pair_mode="fixed2",    # ['random2','fixed2'] pick the 2 heads used in TD targets when n_Q>2
			q_actor_agg="auto",             # ['auto','min2','mean'] how to aggregate Q_pi in actor loss when n_Q>2
			tea_oracle_path=None,           # optional expert actor checkpoint (oracle; ablation only)
			tea_oracle_use_for_gating=False, # if True, override gate with oracle decision
			tea_oracle_margin=0.0,          # margin for oracle KL comparison
			tea_minor_coef=0.0,
			# ---- RA-CGR / Support-Mass CGR knobs ----
			tea_ra_eval_mode="mean",        # RA-CGR: evaluate teacher components by mean action or one stochastic sample
			tea_sm_min_count=1,             # Support-Mass: required number of positive teacher candidates; <=0 uses min_frac
			tea_sm_min_frac=0.0,            # Support-Mass: required positive fraction when min_count <= 0
			tea_sm_mean_margin=0.0,         # Support-Mass: required mean positive advantage margin
			tea_sm_coverage_threshold=0.0,  # Support-Mass: optional M-prior coverage threshold; <=0 disables
			tea_sm_eval_mode="sample",      # Support-Mass: evaluate candidates by mean action or one stochastic sample
			# ---- Certified Support-Transfer Routing (CSTR) knobs ----
			tea_cstr_frac=0.25,                # top-rho support fraction; m=ceil(rho*K) candidates define the certificate
			tea_cstr_margin=0.0,               # threshold on top-rho mean advantage
			tea_cstr_pos_margin=0.0,           # threshold for counting positive support mass
			tea_cstr_coverage=0.0,             # require 1-(1-pos_frac)^M >= this value; <=0 disables the hard coverage test
			tea_cstr_require_pos_count=1,      # if true, require at least ceil(rho*K) positive candidates
			tea_cstr_use_ra_veto=0,            # optional local KL-field sanity check via mixture responsibilities
			tea_cstr_ra_margin=0.0,            # RA veto passes if RA advantage > -margin
			tea_cstr_eval_mode="sample",     # evaluate screening candidates by target policy mean or stochastic sample
			tea_cstr_policy_q="same",        # q_pi side for legacy q-value certificate: same, ucb, max, mean
			tea_cstr_mean_delta_min=-1e9,     # CSTR-SQ: require mean(delta_all) > this value; very negative disables
			tea_cstr_prior_source="independent", # ['independent','samek_all','samek_subset']
			tea_cstr_prior_subset_m=10,          # when prior_source=samek_subset, randomly choose this many prior components from the same K support pool
			# ---- CSTR local density-field certificate ----
			tea_cstr_use_field_veto=0,        # if true, require local reverse-KL density field to be value-consistent
			tea_cstr_field_lambda=1.0,        # lambda in Delta_loc + lambda * U_rho > margin
			tea_cstr_field_margin=0.0,        # margin for the field veto
			tea_cstr_field_mode="sum",       # ['sum','strict']; sum uses Delta_loc+lambda*U_rho, strict uses Delta_loc only
			tea_cstr_field_eval_mode="mean", # representative component action: target mean or one stochastic sample
			# ---- CSTR critic calibration ----
			tea_gate_critic="online",        # ['online','target']; target is a low-cost EMA/target gate critic
			tea_calib_adv_mode="q_value",    # ['q_value','adv_mean','adv_lcb','adv_min']
			tea_calib_beta=1.0,              # beta for advantage-LCB
			tea_calib_penalty=0.0,           # constant calibration penalty subtracted from lower advantages
			tea_calib_td_coef=0.0,           # multiplier for running TD-residual calibration penalty
			tea_calib_td_quantile=0.8,       # quantile of batch TD residual used for running calibration
			tea_calib_td_ema_beta=0.98,      # EMA beta for TD-residual calibration
			tea_calib_td_clip=10.0,          # clip dynamic TD penalty; <=0 disables clipping
			# ---- Diagnostics / reproducibility ----
			tea_diag_every=200,             # log diagnostics every N training steps
			tea_diag_eps=1e-3,              # "near-zero" threshold for Delta
			tea_diag_ema_beta=0.95,         # EMA smoothing for correlation stats
			tea_eval_corr_window=25,        # number of eval points for correlation
			# ---- Choose+ (conservative hard routing) ----
			tea_chooseplus_k_low=0.0,        # z-score lower threshold (<=k_low -> RL)
			tea_chooseplus_k_high=1.0,       # z-score upper threshold (>=k_high -> imitate)
			tea_chooseplus_eps=1e-8,         # numerical stability for z-score
			# ---- Diagnostics: gradient conflict + calibration ----
			tea_diag_bins=10,                # number of quantile bins for delta calibration
			tea_diag_log_hist=True,          # log histograms for delta / gate
			tea_diag_log_grad=True,          # log gradient conflict diagnostics
			# ---- SPReD fidelity knobs ----
			spred_strict=True,            # if True and tea_mode in spred*, match SPReD-like gradient/ensemble details
			spred_detach_w=False,         # if True, detach SPReD weights from actor gradients (stability; deviates from source)
			spred_policy_noise=0.2,       # TD3 target policy smoothing noise (SPReD)
			spred_noise_clip=0.5,          # TD3 noise clip (SPReD)
			spred_policy_freq=2,           # delayed policy updates (SPReD)
			spred_lambda1=1.0,             # SPReD: weight on RL term (Q)
			spred_lambda2=1.0,             # SPReD: weight on imitation term (BC)
			grad_clip_norm=0.0,           # if >0, clip actor/critic/subgoal grad norm
			writer=None
		):
		# Actor
		self.actor = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
		self.actor_target = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())

		# Critic (ensemble over Q networks)
		self.critic_n_q = int(critic_n_q)
		self.critic = EnsembleCritic(state_dim, action_dim, n_Q=self.critic_n_q).to(device)
		self.critic_target = EnsembleCritic(state_dim, action_dim, n_Q=self.critic_n_q).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=q_lr)

		# Subgoal policy
		self.subgoal_net = LaplacePolicy(state_dim).to(device)
		self.subgoal_optimizer = torch.optim.Adam(self.subgoal_net.parameters(), lr=h_lr)

		# Encoder (for vision-based envs)
		self.image_env = image_env
		if self.image_env:
			self.encoder = Encoder(state_dim=state_dim).to(device)
			self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=enc_lr)

		# Actor-Critic Hyperparameters
		self.tau = tau
		self.target_update_interval = target_update_interval
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

		# High-level policy hyperparameters
		self.Lambda = Lambda
		self.n_ensemble = n_ensemble

		# ---- TEA-RIS hyperparameters ----
		self.tea_enabled = bool(tea_enabled)
		self.tea_mode = str(tea_mode)
		assert self.tea_mode in ["gated_kl", "choose", "choose_plus", "ra_cgr", "support_mass", "cstr", "ensqfilter", "spredp", "sprede"], (
			"tea_mode must be one of: 'gated_kl', 'choose', 'choose_plus', 'ra_cgr', 'support_mass', 'cstr', 'ensqfilter', 'spredp', 'sprede'"
		)
		# Keep RIS baseline modes unchanged: require n_Q=2 outside SPReD family.
		if (self.tea_mode not in ["ensqfilter", "spredp", "sprede"]) and int(critic_n_q) != 2:
			raise ValueError(
				f"For tea_mode='{self.tea_mode}', this codebase keeps the original RIS/TEA update unchanged and requires critic_n_q=2 (got {critic_n_q})."
			)
		if (self.tea_mode in ["ensqfilter", "spredp", "sprede"]) and int(critic_n_q) < 2:
			raise ValueError(f"For tea_mode='{self.tea_mode}', critic_n_q must be >= 2 (got {critic_n_q}).")
		self.tea_q_beta = float(tea_q_beta)
		self.tea_margin = float(tea_margin)
		self.tea_temp = float(tea_temp)
		self.tea_hard_gate = bool(tea_hard_gate)
		self.tea_best_of = int(tea_best_of)
		# Q-filtered KL prior
		self.tea_prior_mode = str(tea_prior_mode)
		assert self.tea_prior_mode in ["all", "q_topm"], "tea_prior_mode must be 'all' or 'q_topm'"
		self.tea_prior_top_m = int(tea_prior_top_m)
		self.tea_prior_tau = float(tea_prior_tau)
		self.tea_prior_q_temp = float(tea_prior_q_temp)
		self.tea_prior_use_bestofk = bool(tea_prior_use_bestofk)
		self.tea_prior_eval_mode = str(tea_prior_eval_mode)
		assert self.tea_prior_eval_mode in ["mean", "sample"], "tea_prior_eval_mode must be 'mean' or 'sample'"
		self.tea_warmup_steps = int(tea_warmup_steps)
		self.tea_q_mode = str(tea_q_mode)
		assert self.tea_q_mode in ["min", "lcb"], "tea_q_mode must be 'min' or 'lcb'"

		# ---- Ensemble aggregation controls ----
		self.q_target_pair_mode = str(q_target_pair_mode)
		assert self.q_target_pair_mode in ["random2", "fixed2"], (
			"q_target_pair_mode must be 'random2' or 'fixed2'"
		)
		self.q_actor_agg = str(q_actor_agg)
		assert self.q_actor_agg in ["auto", "min2", "mean"], (
			"q_actor_agg must be 'auto', 'min2', or 'mean'"
		)

		# ---- SPReD fidelity knobs ----
		self.spred_strict = bool(spred_strict)
		self.spred_detach_w = bool(spred_detach_w)
		self.spred_policy_noise = float(spred_policy_noise)
		self.spred_noise_clip = float(spred_noise_clip)
		self.spred_policy_freq = int(spred_policy_freq)
		self.spred_lambda1 = float(spred_lambda1)
		self.spred_lambda2 = float(spred_lambda2)
		self.grad_clip_norm = float(grad_clip_norm)
		if self.spred_strict and (self.tea_mode in ["ensqfilter", "spredp", "sprede"]) and self.critic_n_q > 2:
			# SPReD source: actor uses mean-of-ensemble Q; TD target uses min over a random subset of 2 critics
			self.q_actor_agg = "mean"
			self.q_target_pair_mode = "random2"

		# ---- Optional oracle (expert) gating for analysis/ablation ----
		self.tea_oracle_path = tea_oracle_path
		self.tea_oracle_use_for_gating = bool(tea_oracle_use_for_gating)
		self.tea_oracle_margin = float(tea_oracle_margin)
		self.tea_minor_coef = float(tea_minor_coef)
		# ---- RA-CGR / Support-Mass CGR configuration ----
		self.tea_ra_eval_mode = str(tea_ra_eval_mode)
		assert self.tea_ra_eval_mode in ["mean", "sample"], "tea_ra_eval_mode must be 'mean' or 'sample'"
		self.tea_sm_min_count = int(tea_sm_min_count)
		self.tea_sm_min_frac = float(tea_sm_min_frac)
		self.tea_sm_mean_margin = float(tea_sm_mean_margin)
		self.tea_sm_coverage_threshold = float(tea_sm_coverage_threshold)
		self.tea_sm_eval_mode = str(tea_sm_eval_mode)
		assert self.tea_sm_eval_mode in ["mean", "sample"], "tea_sm_eval_mode must be 'mean' or 'sample'"
		# ---- CSTR configuration ----
		self.tea_cstr_frac = float(tea_cstr_frac)
		self.tea_cstr_margin = float(tea_cstr_margin)
		self.tea_cstr_pos_margin = float(tea_cstr_pos_margin)
		self.tea_cstr_coverage = float(tea_cstr_coverage)
		self.tea_cstr_require_pos_count = bool(tea_cstr_require_pos_count)
		self.tea_cstr_use_ra_veto = bool(tea_cstr_use_ra_veto)
		self.tea_cstr_ra_margin = float(tea_cstr_ra_margin)
		self.tea_cstr_eval_mode = str(tea_cstr_eval_mode)
		assert self.tea_cstr_eval_mode in ["mean", "sample"], "tea_cstr_eval_mode must be 'mean' or 'sample'"
		self.tea_cstr_policy_q = str(tea_cstr_policy_q)
		assert self.tea_cstr_policy_q in ["same", "ucb", "max", "mean"], "tea_cstr_policy_q must be one of: same, ucb, max, mean"
		self.tea_cstr_mean_delta_min = float(tea_cstr_mean_delta_min)
		self.tea_cstr_prior_source = str(tea_cstr_prior_source)
		assert self.tea_cstr_prior_source in ["independent", "samek_all", "samek_subset"], "tea_cstr_prior_source must be one of: independent, samek_all, samek_subset"
		self.tea_cstr_prior_subset_m = int(tea_cstr_prior_subset_m)
		if self.tea_cstr_prior_subset_m < 1:
			raise ValueError(f"tea_cstr_prior_subset_m must be >=1, got {self.tea_cstr_prior_subset_m}")
		# ---- Local density-field certificate ----
		self.tea_cstr_use_field_veto = bool(tea_cstr_use_field_veto)
		self.tea_cstr_field_lambda = float(tea_cstr_field_lambda)
		self.tea_cstr_field_margin = float(tea_cstr_field_margin)
		self.tea_cstr_field_mode = str(tea_cstr_field_mode)
		assert self.tea_cstr_field_mode in ["sum", "strict"], "tea_cstr_field_mode must be 'sum' or 'strict'"
		self.tea_cstr_field_eval_mode = str(tea_cstr_field_eval_mode)
		assert self.tea_cstr_field_eval_mode in ["mean", "sample"], "tea_cstr_field_eval_mode must be 'mean' or 'sample'"
		# ---- Critic calibration used by CSTR certificates ----
		self.tea_gate_critic = str(tea_gate_critic)
		assert self.tea_gate_critic in ["online", "target"], "tea_gate_critic must be 'online' or 'target'"
		self.tea_calib_adv_mode = str(tea_calib_adv_mode)
		assert self.tea_calib_adv_mode in ["q_value", "adv_mean", "adv_lcb", "adv_min"], "tea_calib_adv_mode must be one of: q_value, adv_mean, adv_lcb, adv_min"
		self.tea_calib_beta = float(tea_calib_beta)
		self.tea_calib_penalty = float(tea_calib_penalty)
		self.tea_calib_td_coef = float(tea_calib_td_coef)
		self.tea_calib_td_quantile = float(tea_calib_td_quantile)
		self.tea_calib_td_ema_beta = float(tea_calib_td_ema_beta)
		self.tea_calib_td_clip = float(tea_calib_td_clip)
		self._calib_td_ema = 0.0
		self._calib_td_last = 0.0
		if not (0.0 <= self.tea_calib_td_quantile <= 1.0):
			raise ValueError(f"tea_calib_td_quantile must be in [0,1], got {self.tea_calib_td_quantile}")
		if not (0.0 < self.tea_cstr_frac <= 1.0):
			raise ValueError(f"tea_cstr_frac must be in (0,1], got {self.tea_cstr_frac}")
		if self.tea_cstr_coverage > 1.0:
			raise ValueError(f"tea_cstr_coverage must be <= 1, got {self.tea_cstr_coverage}")
		if self.tea_mode == "ra_cgr" and self.tea_prior_mode != "all":
			raise ValueError("RA-CGR requires tea_prior_mode=all so that the gate is aligned with the random covering prior used by D_KL.")
		if self.tea_mode == "cstr" and self.tea_cstr_use_ra_veto and self.tea_prior_mode != "all":
			raise ValueError("CSTR with RA veto requires tea_prior_mode=all so that the veto is aligned with the random covering prior used by D_KL.")
		if self.tea_mode == "cstr" and self.tea_cstr_use_field_veto and self.tea_prior_mode != "all":
			raise ValueError("CSTR field veto requires tea_prior_mode=all because it uses responsibilities of the actual random covering prior.")
		self.expert_actor = None
		if self.tea_oracle_path is not None and str(self.tea_oracle_path) != "":
			if not os.path.isfile(str(self.tea_oracle_path)):
				raise FileNotFoundError(f"[TEA-RIS] tea_oracle_path not found: {self.tea_oracle_path}")
			self.expert_actor = GaussianPolicy(state_dim, action_dim).to(device)
			self.expert_actor.load_state_dict(torch.load(str(self.tea_oracle_path), map_location=device))
			self.expert_actor.eval()
			for p in self.expert_actor.parameters():
				p.requires_grad = False

		# Utils
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.logger = logger
		self.writer = writer
		self.total_it = 0

		# ---- Diagnostics buffers (for "why it works" analyses) ----
		self.tea_diag_every = int(tea_diag_every)
		self.tea_diag_eps = float(tea_diag_eps)
		self.tea_diag_ema_beta = float(tea_diag_ema_beta)
		self.tea_eval_corr_window = int(tea_eval_corr_window)
		# Choose+
		self.tea_chooseplus_k_low = float(tea_chooseplus_k_low)
		self.tea_chooseplus_k_high = float(tea_chooseplus_k_high)
		self.tea_chooseplus_eps = float(tea_chooseplus_eps)
		# Diagnostics
		self.tea_diag_bins = int(tea_diag_bins)
		self.tea_diag_log_hist = bool(tea_diag_log_hist)
		self.tea_diag_log_grad = bool(tea_diag_log_grad)

		self._diag_ema = {}          # name -> float
		self._diag_last = {}         # last-step snapshot
		self._eval_hist = {}         # tag -> list of dict snapshots

		# ---- Compact direct-to-disk diagnostics (JSONL; avoids TensorBoard event export) ----
		self.compact_log_enabled = False
		self.compact_log_dir = None
		self.compact_log_every = int(self.tea_diag_every)
		self.compact_log_flush_every = 1
		self._compact_train_fp = None
		self._compact_eval_fp = None
		self._compact_rows_since_flush = 0

	def set_compact_logger(self, run_folder, run_meta=None, log_every=None, flush_every=1, dirname="compact_logs"):
		"""Enable compact JSONL diagnostics written directly during training.

		This logger is intentionally independent of TensorBoard event files.  It
		writes two small files under ``run_folder/compact_logs``:
		  - train_diag.jsonl: one aggregated train/route diagnostic row every N updates;
		  - eval.jsonl: one row per random/farthest evaluation.
		"""
		try:
			log_dir = os.path.join(str(run_folder), str(dirname))
			os.makedirs(log_dir, exist_ok=True)
			self.compact_log_enabled = True
			self.compact_log_dir = log_dir
			self.compact_log_every = int(log_every if log_every is not None and int(log_every) > 0 else self.tea_diag_every)
			self.compact_log_every = max(1, self.compact_log_every)
			self.compact_log_flush_every = int(max(1, flush_every))
			self._compact_train_fp = open(os.path.join(log_dir, "train_diag.jsonl"), "a", encoding="utf-8", buffering=1)
			self._compact_eval_fp = open(os.path.join(log_dir, "eval.jsonl"), "a", encoding="utf-8", buffering=1)
			meta = {
				"created_at_wall_time": time.time(),
				"compact_log_every": self.compact_log_every,
				"compact_log_flush_every": self.compact_log_flush_every,
				"run_meta": run_meta or {},
			}
			with open(os.path.join(log_dir, "compact_meta.json"), "w", encoding="utf-8") as f:
				json.dump(meta, f, indent=2, sort_keys=True)
		except Exception as exc:
			print(f"[WARN] failed to enable compact logger: {exc}", flush=True)
			self.compact_log_enabled = False

	def _compact_float(self, value):
		"""Convert scalars/tensors to JSON-safe floats; return None for non-finite values."""
		try:
			if torch.is_tensor(value):
				v = value.detach().float()
				if v.numel() == 0:
					return None
				out = float(v.mean().item())
			else:
				out = float(value)
			if not np.isfinite(out):
				return None
			return out
		except Exception:
			return None

	def _compact_tensor_stats(self, rec, prefix, value, include_quantile=False):
		"""Append mean/std/min/max/(optional p95) stats for a tensor-like value."""
		try:
			if value is None:
				return
			if not torch.is_tensor(value):
				v = torch.as_tensor(value)
			else:
				v = value.detach()
			v = v.float().reshape(-1)
			if v.numel() == 0:
				return
			rec[f"{prefix}_mean"] = self._compact_float(v.mean())
			rec[f"{prefix}_std"] = self._compact_float(v.std(unbiased=False))
			rec[f"{prefix}_min"] = self._compact_float(v.min())
			rec[f"{prefix}_max"] = self._compact_float(v.max())
			if include_quantile and v.numel() >= 4:
				rec[f"{prefix}_p95"] = self._compact_float(torch.quantile(v, 0.95))
		except Exception:
			return

	def _compact_write_jsonl(self, fp, rec):
		if (not self.compact_log_enabled) or fp is None:
			return
		try:
			# Drop None-only keys?  Keep them: pandas will parse them as NaN.
			fp.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")
			self._compact_rows_since_flush += 1
			if self._compact_rows_since_flush >= self.compact_log_flush_every:
				fp.flush()
				self._compact_rows_since_flush = 0
		except Exception:
			return

	def _compact_log_train(self, t, critic_loss, actor_loss, D_KL, Q_pi, w, delta, q_teach, q_pi_eval, gate_info=None, kl_info=None):
		"""Write one compact training diagnostic row."""
		if not self.compact_log_enabled:
			return
		try:
			rec = {
				"split": "train",
				"step": int(t),
				"total_it": int(self.total_it),
				"wall_time": float(time.time()),
				"tea_mode": str(self.tea_mode),
				"tea_best_of": int(self.tea_best_of),
				"n_ensemble": int(self.n_ensemble),
				"tea_cstr_prior_source": str(getattr(self, "tea_cstr_prior_source", "independent")),
				"tea_cstr_eval_mode": str(getattr(self, "tea_cstr_eval_mode", "")),
				"tea_cstr_use_field_veto": int(bool(getattr(self, "tea_cstr_use_field_veto", False))),
				"tea_cstr_field_eval_mode": str(getattr(self, "tea_cstr_field_eval_mode", "")),
				"tea_gate_critic": str(getattr(self, "tea_gate_critic", "online")),
				"tea_calib_adv_mode": str(getattr(self, "tea_calib_adv_mode", "q_value")),
				"alpha": float(self.alpha),
				"tea_minor_coef": float(self.tea_minor_coef),
				"actor_loss": self._compact_float(actor_loss),
				"critic_loss": self._compact_float(critic_loss),
				"calib_penalty": self._compact_float(self._calibration_penalty_value()),
				"td_residual_last": self._compact_float(getattr(self, "_calib_td_last", 0.0)),
				"td_residual_ema": self._compact_float(getattr(self, "_calib_td_ema", 0.0)),
			}
			if kl_info is not None and isinstance(kl_info, dict):
				for k in ["prior_source", "support_pool_size", "prior_pool_size"]:
					if k in kl_info:
						try:
							rec[f"kl_{k}"] = int(kl_info[k]) if isinstance(kl_info[k], (int, np.integer)) else str(kl_info[k])
						except Exception:
							rec[f"kl_{k}"] = str(kl_info[k])
			self._compact_tensor_stats(rec, "KL", D_KL, include_quantile=True)
			self._compact_tensor_stats(rec, "Q_pi", Q_pi, include_quantile=False)
			self._compact_tensor_stats(rec, "gate", w, include_quantile=False)
			self._compact_tensor_stats(rec, "delta", delta, include_quantile=True)
			self._compact_tensor_stats(rec, "q_teach", q_teach, include_quantile=False)
			self._compact_tensor_stats(rec, "q_pi_eval", q_pi_eval, include_quantile=False)
			if gate_info is not None:
				for key, value in gate_info.items():
					if torch.is_tensor(value):
						rec[f"{key}_mean"] = self._compact_float(value)
					else:
						rec[str(key)] = self._compact_float(value)
			self._compact_write_jsonl(self._compact_train_fp, rec)
		except Exception:
			return

	def _compact_log_eval(self, tag, t, success_rate, eval_distance=None):
		"""Write one compact evaluation row."""
		if not self.compact_log_enabled:
			return
		try:
			rec = {
				"split": "eval",
				"tag": str(tag),
				"step": int(t),
				"wall_time": float(time.time()),
				"success": self._compact_float(success_rate),
				"eval_distance": self._compact_float(eval_distance),
				"delta_ema": self._compact_float(self._diag_ema.get("delta", np.nan)),
				"gate_ema": self._compact_float(self._diag_ema.get("gate", np.nan)),
				"tea_cstr_prior_source": str(getattr(self, "tea_cstr_prior_source", "independent")),
				"tea_cstr_eval_mode": str(getattr(self, "tea_cstr_eval_mode", "")),
				"tea_cstr_use_field_veto": int(bool(getattr(self, "tea_cstr_use_field_veto", False))),
				"tea_gate_critic": str(getattr(self, "tea_gate_critic", "online")),
				"tea_calib_adv_mode": str(getattr(self, "tea_calib_adv_mode", "q_value")),
			}
			self._compact_write_jsonl(self._compact_eval_fp, rec)
		except Exception:
			return

	def save(self, folder, save_optims=False):
		torch.save(self.actor.state_dict(), folder + "actor.pth")
		torch.save(self.critic.state_dict(), folder + "critic.pth")
		torch.save(self.subgoal_net.state_dict(), folder + "subgoal_net.pth")
		if self.image_env:
			torch.save(self.encoder.state_dict(), folder + "encoder.pth")
		if save_optims:
			torch.save(self.actor_optimizer.state_dict(), folder + "actor_opti.pth")
			torch.save(self.critic_optimizer.state_dict(), folder + "critic_opti.pth")
			torch.save(self.subgoal_optimizer.state_dict(), folder + "subgoal_opti.pth")
			if self.image_env:
				torch.save(self.encoder_optimizer.state_dict(), folder + "encoder_opti")

	def load(self, folder):
		self.actor.load_state_dict(torch.load(folder + "actor.pth", map_location=self.device))
		self.critic.load_state_dict(torch.load(folder + "critic.pth", map_location=self.device))
		self.subgoal_net.load_state_dict(torch.load(folder + "subgoal_net.pth", map_location=self.device))
		if self.image_env:
			self.encoder.load_state_dict(torch.load(folder + "encoder.pth", map_location=self.device))

	@torch.no_grad()
	def select_action(self, state, goal):
		state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
		goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
		if self.image_env:
			state = state.view(1, 3, 84, 84)
			goal = goal.view(1, 3, 84, 84)
			state = self.encoder(state)
			goal = self.encoder(goal)
		action, _, _ = self.actor.sample(state, goal)
		return action.cpu().data.numpy().flatten()

	def _q_min2(self, q, idx2=None):
		"""TD3-style conservative reduction for an ensemble Q.

		Why this exists:
		- Original RIS uses n_Q=2 and takes min over both heads.
		- SPReD-style baselines often use n_Q=10 for uncertainty.
		  If we naively take min over *all* 10 heads, targets/updates become overly
		  pessimistic and training collapses.

		For n_Q>2 we therefore take min over a subset of 2 heads (as SPReD does).

		Args:
			q:    [B, n_Q]
			idx2: [2] indices for the chosen heads; if None, sample 2 globally.
		Returns:
			q_min: [B,1]
		"""
		if q.size(-1) <= 2:
			return q.min(-1, keepdim=True)[0]
		if idx2 is None:
			idx2 = torch.randperm(q.size(-1), device=q.device)[:2]
		q2 = q.index_select(-1, idx2)
		return q2.min(-1, keepdim=True)[0]

	def value(self, state, goal, idx2=None):
		_, _, action = self.actor.sample(state, goal)
		q = self.critic(state, action, goal)
		V = self._q_min2(q, idx2=idx2)
		return V

	def sample_subgoal_n(self, state, goal, n):
		"""Sample exactly n subgoals from the current high-level policy."""
		n = int(max(1, n))
		subgoal_distribution = self.subgoal_net(state, goal)
		subgoal = subgoal_distribution.rsample((n,))
		subgoal = torch.transpose(subgoal, 0, 1)
		return subgoal

	def sample_subgoal(self, state, goal):
		return self.sample_subgoal_n(state, goal, self.n_ensemble)

	def _critic_module(self, use_gate_critic=False):
		"""Return the critic module used for a value query.

		CSTR calibration can use the online critic or the slowly moving target
		critic.  The target critic acts as a cheap EMA gate critic and reduces
		self-exploitation in routing decisions.  Non-CSTR legacy calls keep using
		the online critic by default.
		"""
		if bool(use_gate_critic) and getattr(self, "tea_gate_critic", "online") == "target":
			return self.critic_target
		return self.critic

	def _critic_q_all(self, state, action, goal, use_gate_critic=False):
		critic = self._critic_module(use_gate_critic=use_gate_critic)
		return critic(state, action, goal)

	def _critic_q_stats(self, state, action, goal, idx2=None, use_gate_critic=False):
		"""Return (q_min, q_mean, q_std) with shape [B,1].

		q_min uses TD3-style min2 when n_Q>2; mean/std use all heads.
		"""
		q = self._critic_q_all(state, action, goal, use_gate_critic=use_gate_critic)  # [B, n_Q]
		q_min = self._q_min2(q, idx2=idx2)
		q_mean = q.mean(-1, keepdim=True)
		# std over ensemble heads; n_Q is small (default=2), still useful as a stability heuristic
		q_std = q.std(-1, keepdim=True, unbiased=False)
		return q_min, q_mean, q_std

	def _critic_q_eval(self, state, action, goal, idx2=None, use_gate_critic=False):
		"""Conservative evaluation used in gating Delta.

		- 'min':  q_min
		- 'lcb':  q_mean - beta * q_std
		"""
		q_min, q_mean, q_std = self._critic_q_stats(state, action, goal, idx2=idx2, use_gate_critic=use_gate_critic)
		if self.tea_q_mode == "min":
			return q_min
		return q_mean - self.tea_q_beta * q_std

	def _critic_q_eval_named(self, state, action, goal, mode="same", idx2=None, use_gate_critic=False):
		"""Evaluate Q with a named aggregation used by CSTR certificates.

		CSTR is most conservative when the teacher side uses a lower-confidence
		estimate and the policy side uses an upper-confidence estimate. For
		backward-compatible ablations, mode='same' reuses _critic_q_eval.
		"""
		mode = str(mode)
		if mode == "same":
			return self._critic_q_eval(state, action, goal, idx2=idx2, use_gate_critic=use_gate_critic)
		q = self._critic_q_all(state, action, goal, use_gate_critic=use_gate_critic)
		if mode == "max":
			return q.max(-1, keepdim=True)[0]
		q_mean = q.mean(-1, keepdim=True)
		if mode == "mean":
			return q_mean
		if mode == "ucb":
			q_std = q.std(-1, keepdim=True, unbiased=False)
			return q_mean + self.tea_q_beta * q_std
		raise ValueError(f"Unknown Q eval mode: {mode}")

	def _calibration_penalty_value(self):
		"""Scalar penalty b_t used in calibrated lower advantages.

		b_t = constant penalty + td_coef * EMA(batch TD residual quantile).
		The TD term is optional and defaults to zero; it is exposed as a
		hyperparameter because early experiments should first isolate field vs.
		calibration effects.
		"""
		td_term = float(getattr(self, "_calib_td_ema", 0.0))
		pen = float(getattr(self, "tea_calib_penalty", 0.0)) + float(getattr(self, "tea_calib_td_coef", 0.0)) * td_term
		if getattr(self, "tea_calib_td_clip", 0.0) and float(self.tea_calib_td_clip) > 0.0:
			pen = min(pen, float(self.tea_calib_td_clip))
		return float(max(0.0, pen))

	@torch.no_grad()
	def _update_td_calibration(self, current_Q, target_Q):
		"""Update the running TD-residual calibration statistic.

		current_Q: [B,n_Q] online critic values before/around the critic update.
		target_Q:  [B,1] TD target used by the critic update.
		"""
		try:
			resid = (current_Q.detach() - target_Q.detach()).abs().mean(dim=-1)
			q = torch.quantile(resid.float(), float(self.tea_calib_td_quantile)).item()
			if self.tea_calib_td_clip and float(self.tea_calib_td_clip) > 0.0:
				q = min(float(q), float(self.tea_calib_td_clip))
			self._calib_td_last = float(q)
			beta = float(self.tea_calib_td_ema_beta)
			self._calib_td_ema = beta * float(getattr(self, "_calib_td_ema", 0.0)) + (1.0 - beta) * float(q)
		except Exception:
			return

	def _candidate_lower_advantages(self, state, goal, a_pi, a_cand, use_gate_critic=True):
		"""Evaluate candidate lower advantages for CSTR certificates.

		Args:
			state, goal: [B,state_dim]
			a_pi:       [B,action_dim] current actor action, detached by caller
			a_cand:     [B,K,action_dim] candidate teacher/prior actions

		Returns:
			delta_all: [B,K] lower advantages ell_k
			q_cand_summary: [B,K] mean candidate Q for logging
			q_pi_eval: [B,1] policy-side Q summary for logging
			adv_std: [B,K] per-candidate advantage disagreement for diagnostics
		"""
		B = state.size(0)
		K = int(a_cand.size(1))
		state_rep = state.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
		goal_rep = goal.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
		a_flat = a_cand.reshape(B * K, self.action_dim)
		mode = str(getattr(self, "tea_calib_adv_mode", "q_value"))
		penalty = self._calibration_penalty_value()
		if mode == "q_value":
			q_cand = self._critic_q_eval(state_rep, a_flat, goal_rep, use_gate_critic=use_gate_critic).reshape(B, K)
			q_pi_eval = self._critic_q_eval_named(state, a_pi.detach(), goal, mode=self.tea_cstr_policy_q, use_gate_critic=use_gate_critic)
			delta_all = q_cand - q_pi_eval - float(penalty)
			q_cand_summary = q_cand
			adv_std = torch.zeros_like(delta_all)
			return delta_all, q_cand_summary, q_pi_eval, adv_std

		q_cand_heads = self._critic_q_all(state_rep, a_flat, goal_rep, use_gate_critic=use_gate_critic).reshape(B, K, -1)
		q_pi_heads = self._critic_q_all(state, a_pi.detach(), goal, use_gate_critic=use_gate_critic)  # [B,n_Q]
		delta_heads = q_cand_heads - q_pi_heads.unsqueeze(1)  # [B,K,n_Q]
		delta_mean = delta_heads.mean(dim=-1)
		adv_std = delta_heads.std(dim=-1, unbiased=False)
		if mode == "adv_mean":
			delta_all = delta_mean - float(penalty)
		elif mode == "adv_lcb":
			delta_all = delta_mean - float(self.tea_calib_beta) * adv_std - float(penalty)
		elif mode == "adv_min":
			delta_all = delta_heads.min(dim=-1)[0] - float(penalty)
		else:
			raise ValueError(f"Unknown tea_calib_adv_mode: {mode}")
		q_cand_summary = q_cand_heads.mean(dim=-1)
		q_pi_eval = q_pi_heads.mean(dim=-1, keepdim=True)
		return delta_all, q_cand_summary, q_pi_eval, adv_std

	def _sample_teacher_action_bestofk(self, state, goal, K, idx2=None):
		"""Sample teacher actions via subgoals and return best-of-K by conservative Q.

		Teacher action candidates are produced by:
		- sample z_k ~ subgoal_net(s,g)
		- sample a_k ~ actor_target(s, z_k)
		and evaluated using Q(s, a_k, g) (note goal is original g).

		Returns:
			a_teach_best: [B, action_dim]
			q_teach_best: [B, 1]
		"""
		B = state.size(0)
		with torch.no_grad():
			K = int(max(1, K))
			subgoals = self.sample_subgoal_n(state, goal, K)  # [B, K, state_dim]

			# Sample candidate actions from target actor conditioned on subgoals
			state_rep = state.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
			sub_flat = subgoals.reshape(B * K, self.state_dim)
			a_flat, _, _ = self.actor_target.sample(state_rep, sub_flat)  # tanh actions
			a_cand = a_flat.reshape(B, K, self.action_dim)

			# Evaluate candidate actions under original goal
			goal_rep = goal.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
			q_cand = self._critic_q_eval(state_rep, a_flat, goal_rep, idx2=idx2).reshape(B, K, 1)
			q_best, idx = q_cand.max(dim=1)  # [B,1], idx: [B,1]

			idx = idx.view(B, 1, 1).expand(B, 1, self.action_dim)
			a_best = a_cand.gather(dim=1, index=idx).squeeze(1)  # [B, action_dim]
			return a_best, q_best

	def _diag_normal_kl(self, dist_p, dist_q):
		"""KL(dist_p || dist_q) for diagonal Gaussians (Normal) with shape [B, action_dim]."""
		mu_p, std_p = dist_p.loc, dist_p.scale
		mu_q, std_q = dist_q.loc, dist_q.scale
		var_p = std_p.pow(2)
		var_q = std_q.pow(2)
		# 0.5 * sum( log(var_q/var_p) + (var_p + (mu_p-mu_q)^2)/var_q - 1 )
		kl = 0.5 * (torch.log(var_q + self.epsilon) - torch.log(var_p + self.epsilon) + (var_p + (mu_p - mu_q).pow(2)) / (var_q + self.epsilon) - 1.0)
		return kl.sum(-1, keepdim=True)

	@torch.no_grad()
	def _oracle_gate_from_expert(self, state, goal, K):
		"""Oracle gate using an expert policy (for analysis/ablation only).

		We compare which action distribution is closer to the expert:
		- student policy:      pi(a|s,g)
		- teacher candidates:  pi_targ(a|s, sg_k),  sg_k ~ pi_H(s,g)

		We approximate the tanh-squashed Gaussian by the underlying Normal and compute
		KL(N_student || N_expert) and min_k KL(N_teacher_k || N_expert).

		Gate rule (hard):
			w_oracle = I[ KL_teacher + margin < KL_student ]

		Returns:
			w_oracle:   [B,1] in {0,1}
			kl_student: [B,1]
			kl_teacher: [B,1] (min over K candidates)
		"""
		if self.expert_actor is None:
			return None, None, None

		# Student vs expert
		dist_student = self.actor(state, goal)
		dist_expert = self.expert_actor(state, goal)
		kl_student = self._diag_normal_kl(dist_student, dist_expert)  # [B,1]

		# Teacher candidates vs expert (best-of-K by closeness)
		K = int(max(1, K))
		subgoals = self.sample_subgoal_n(state, goal, K)  # [B, K, state_dim]

		B = state.size(0)
		state_rep = state.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
		sg_flat = subgoals.reshape(B * K, self.state_dim)

		dist_teacher = self.actor_target(state_rep, sg_flat)  # Normal with shape [B*K, action_dim]
		dist_expert_rep = self.expert_actor(state_rep, goal.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim))
		kl_teacher_all = self._diag_normal_kl(dist_teacher, dist_expert_rep).reshape(B, K, 1)
		kl_teacher, _ = kl_teacher_all.min(dim=1)  # [B,1]

		w_oracle = (kl_teacher + self.tea_oracle_margin < kl_student).float()
		return w_oracle, kl_student, kl_teacher

	def sample_action_and_KL(self, state, goal, return_info=False, subgoal=None):
		"""Sample an actor action and compute the RIS single-sample KL estimator.

		D_KL is computed as:
			log pi_theta(u|s,g) - log rho(u|s,g),
		where u is the pre-tanh actor sample and rho is the subgoal-induced
		pre-tanh teacher mixture built from actor_target.

		When return_info=True, this method also returns the exact random covering
		prior components used by D_KL. RA-CGR relies on this to make its routing
		decision responsibility-aligned with the KL vector field.
		"""
		batch_size = state.size(0)
		action_dist = self.actor(state, goal)
		raw_action = action_dist.rsample()  # pre-tanh actor sample

		# Subgoals used to form the KL prior. Keeping a single sampled pool here is
		# important: RA-CGR can then evaluate the same mixture that produced D_KL.
		# For same-K reuse ablations, the caller may pass a pre-sampled support pool
		# or a random subset of it.
		if subgoal is None:
			subgoal = self.sample_subgoal(state, goal)  # [B, M, state_dim]
		kl_info = {
			"raw_action": raw_action,
			"subgoal": subgoal,
			"prior_mode": self.tea_prior_mode,
			"prior_pool_size": int(subgoal.size(1)),
		}

		if self.tea_enabled and (self.tea_prior_mode == "q_topm"):
			prior_log_prob = self._prior_log_prob_q_topm(state, goal, raw_action, subgoal=subgoal)
		else:
			# Original RIS: uniform mixture over all sampled subgoals.
			prior_dist_full = self.actor_target(
				state.unsqueeze(1).expand(batch_size, subgoal.size(1), self.state_dim),
				subgoal
			)
			# Detach prior parameters, while retaining gradients wrt raw_action.
			prior_loc = prior_dist_full.loc.detach()
			prior_scale = prior_dist_full.scale.detach()
			prior_action_dist = D.Normal(prior_loc, prior_scale)
			raw_rep = raw_action.unsqueeze(1).expand(batch_size, subgoal.size(1), self.action_dim)
			logp_components = prior_action_dist.log_prob(raw_rep).sum(-1)  # [B, M]
			prior_log_prob = torch.logsumexp(logp_components, dim=1, keepdim=True) - np.log(float(subgoal.size(1)))

			if return_info:
				# Mixture responsibilities at the actual actor sample. These are exactly
				# the weights appearing in grad_u log rho_M(u). They are diagnostic / gate
				# signals only, so they are detached from actor optimization.
				kl_info.update({
					"prior_mode": "all",
					"prior_subgoals": subgoal.detach(),
					"prior_loc": prior_loc.detach(),
					"prior_scale": prior_scale.detach(),
					"prior_logp_components": logp_components.detach(),
					"responsibilities": torch.softmax(logp_components.detach(), dim=1),
				})

		D_KL = action_dist.log_prob(raw_action).sum(-1, keepdim=True) - prior_log_prob
		action = torch.tanh(raw_action)
		if return_info:
			return action, D_KL, kl_info
		return action, D_KL

	def _prior_log_prob_q_topm(self, state, goal, raw_action, subgoal=None):
		"""Compute log rho(a|s,g) where rho is a sparse, Q-filtered mixture prior.

		We first generate K candidate subgoals z_k ~ p(z|s,g) (from subgoal_net), then:
			1) sample teacher candidate actions a_k ~ pi_target(\\cdot|s,z_k)
			2) evaluate conservative Q(s,a_k,g)
			3) keep a sparse subset (top-M, optionally within a Q-margin tau of the best)
			4) form rho as a weighted mixture of pi_target(\\cdot|s,z_k) over the kept subset.

		This aligns the KL regularizer with the same Q-based criterion used by TEA choose gating.
		"""
		B = state.size(0)
		if subgoal is None:
			subgoal = self.sample_subgoal(state, goal)  # [B, n_ensemble, state_dim]

		K_all = int(subgoal.size(1))
		K = int(min(self.tea_best_of, K_all)) if bool(self.tea_prior_use_bestofk) else K_all
		K = max(1, K)
		sub_cand = subgoal[:, :K, :]  # [B,K,state_dim]

		# 1-2) sample candidate actions and evaluate conservative Q
		state_rep = state.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
		sub_flat = sub_cand.reshape(B * K, self.state_dim)
		with torch.no_grad():
			# Evaluate each candidate component using either its mean action (lower variance)
			# or a single stochastic sample (closer to best-of-K sampling in gating).
			if str(self.tea_prior_eval_mode) == "mean":
				dist = self.actor_target(state_rep, sub_flat)  # Normal over pre-tanh actions
				a_flat = torch.tanh(dist.loc)
			else:
				a_flat, _, _ = self.actor_target.sample(state_rep, sub_flat)  # tanh actions
			goal_rep = goal.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
			q_flat = self._critic_q_eval(state_rep, a_flat, goal_rep).reshape(B, K)  # [B,K]

		# 3) choose a sparse subset
		q_best, idx_best = q_flat.max(dim=1, keepdim=True)  # [B,1], [B,1]
		tau = float(max(0.0, self.tea_prior_tau))
		if tau > 0.0:
			mask = (q_flat >= (q_best - tau))
		else:
			mask = torch.ones_like(q_flat, dtype=torch.bool)

		max_m = int(self.tea_prior_top_m) if int(self.tea_prior_top_m) > 0 else K
		max_m = int(max(1, min(max_m, K)))

		q_for_topk = q_flat.masked_fill(~mask, -1e9)
		top_vals, top_idx = torch.topk(q_for_topk, k=max_m, dim=1)  # [B,max_m]
		valid = (top_vals > -1e8)  # avoid the masked -1e9 placeholders

		# Fallback: if tau masked everything, force-keep the best index
		no_valid = (valid.sum(dim=1, keepdim=True) == 0)
		if no_valid.any():
			# overwrite first slot
			top_idx = top_idx.clone()
			top_vals = top_vals.clone()
			valid = valid.clone()
			top_idx[no_valid.squeeze(1), 0] = idx_best[no_valid.squeeze(1), 0]
			top_vals[no_valid.squeeze(1), 0] = q_best[no_valid.squeeze(1), 0]
			valid[no_valid.squeeze(1), 0] = True

		# 4) mixture weights from Q (detach); normalize over valid entries
		temp = float(self.tea_prior_q_temp)
		q_sel = top_vals  # [B,max_m]
		if temp > 0.0:
			logits = (q_sel - q_sel.max(dim=1, keepdim=True)[0]) / temp
			logits = logits.masked_fill(~valid, -1e9)
			w = torch.softmax(logits, dim=1)
		else:
			w = valid.float()
			w = w / (w.sum(dim=1, keepdim=True) + self.epsilon)

		# Gather selected subgoals
		idx_expand = top_idx.unsqueeze(-1).expand(B, max_m, self.state_dim)
		sub_sel = sub_cand.gather(dim=1, index=idx_expand)  # [B,max_m,state_dim]
		state_sel = state.unsqueeze(1).expand(B, max_m, self.state_dim)

		# Build mixture density at raw_action (detach prior params; keep grad wrt raw_action)
		prior_dist = self.actor_target(state_sel, sub_sel)  # Normal over pre-tanh actions
		prior_dist = D.Normal(prior_dist.loc.detach(), prior_dist.scale.detach())

		raw_rep = raw_action.unsqueeze(1).expand(B, max_m, self.action_dim)
		logp = prior_dist.log_prob(raw_rep).sum(-1)  # [B,max_m]
		logw = torch.log(w + self.epsilon)            # [B,max_m]
		return torch.logsumexp(logp + logw, dim=1, keepdim=True)



	@torch.no_grad()
	def _compute_ra_cgr_gate(self, state, goal, a_pi, kl_info):
		"""Responsibility-Aligned CGR gate.

		RA-CGR evaluates the same random covering prior used by the KL estimator.
		For the pre-tanh actor sample u, the mixture responsibilities r_m(u)
		identify which teacher components actually shape grad log rho_M(u). The
		gate compares a responsibility-weighted conservative teacher value against
		the current policy action value:
			Delta_resp = sum_m r_m Q(s, a_m, g) - Q(s, a_pi, g).
		"""
		if kl_info is None or kl_info.get("prior_mode", None) != "all":
			raise RuntimeError("RA-CGR requires KL info from the random covering prior; use tea_prior_mode=all.")
		if "responsibilities" not in kl_info or "prior_loc" not in kl_info or "prior_scale" not in kl_info:
			raise RuntimeError("RA-CGR missing prior responsibilities/parameters from sample_action_and_KL(return_info=True).")

		B = state.size(0)
		M = int(kl_info["responsibilities"].size(1))
		resp = kl_info["responsibilities"].detach()  # [B,M]

		prior_loc = kl_info["prior_loc"].detach()      # [B,M,A]
		prior_scale = kl_info["prior_scale"].detach()  # [B,M,A]
		if self.tea_ra_eval_mode == "mean":
			a_components = torch.tanh(prior_loc)
		else:
			raw_components = D.Normal(prior_loc, prior_scale).sample()
			a_components = torch.tanh(raw_components)

		state_rep = state.unsqueeze(1).expand(B, M, self.state_dim).reshape(B * M, self.state_dim)
		goal_rep = goal.unsqueeze(1).expand(B, M, self.state_dim).reshape(B * M, self.state_dim)
		a_flat = a_components.reshape(B * M, self.action_dim)
		q_components = self._critic_q_eval(state_rep, a_flat, goal_rep).reshape(B, M)  # [B,M]

		q_resp = (resp * q_components).sum(dim=1, keepdim=True)
		q_best, _ = q_components.max(dim=1, keepdim=True)
		q_mean = q_components.mean(dim=1, keepdim=True)
		q_pi_eval = self._critic_q_eval(state, a_pi.detach(), goal)
		delta = q_resp - q_pi_eval
		w = self._tea_gate(delta)

		pos_mask = (q_components > q_pi_eval).float()
		pos_resp_mass = (resp * pos_mask).sum(dim=1, keepdim=True)
		resp_entropy = -(resp * torch.log(resp + self.epsilon)).sum(dim=1, keepdim=True)
		eff_components = torch.exp(resp_entropy)
		info = {
			"ra_q_resp": q_resp.detach(),
			"ra_q_best": q_best.detach(),
			"ra_q_mean": q_mean.detach(),
			"ra_resp_entropy": resp_entropy.detach(),
			"ra_eff_components": eff_components.detach(),
			"ra_pos_resp_mass": pos_resp_mass.detach(),
		}
		return w.detach(), delta.detach(), q_resp.detach(), q_pi_eval.detach(), info

	@torch.no_grad()
	def _compute_support_mass_gate(self, state, goal, a_pi, K):
		"""Support-Mass CGR gate.

		Instead of firing on a single best-of-K outlier, Support-Mass CGR requires
		multiple sampled teacher components to have positive conservative advantage.
		This is intended to distinguish coverable useful support from rare Q holes.
		"""
		B = state.size(0)
		K = int(max(1, int(K)))
		sub_cand = self.sample_subgoal_n(state, goal, K)  # [B,K,state_dim]

		state_rep = state.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
		sub_flat = sub_cand.reshape(B * K, self.state_dim)
		if self.tea_sm_eval_mode == "mean":
			dist = self.actor_target(state_rep, sub_flat)
			a_flat = torch.tanh(dist.loc)
		else:
			a_flat, _, _ = self.actor_target.sample(state_rep, sub_flat)
		goal_rep = goal.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
		q_cand = self._critic_q_eval(state_rep, a_flat, goal_rep).reshape(B, K)

		q_pi_eval = self._critic_q_eval(state, a_pi.detach(), goal)  # [B,1]
		delta_all = q_cand - q_pi_eval  # [B,K]
		positive = (delta_all > self.tea_margin)
		pos_count = positive.float().sum(dim=1, keepdim=True)
		pos_frac = pos_count / float(K)
		max_delta, _ = delta_all.max(dim=1, keepdim=True)
		mean_delta = delta_all.mean(dim=1, keepdim=True)
		pos_sum = (delta_all * positive.float()).sum(dim=1, keepdim=True)
		pos_mean_delta = pos_sum / (pos_count + self.epsilon)
		pos_mean_delta = torch.where(pos_count > 0.0, pos_mean_delta, max_delta)

		if int(self.tea_sm_min_count) > 0:
			required_count = int(self.tea_sm_min_count)
		else:
			required_count = int(np.ceil(max(0.0, float(self.tea_sm_min_frac)) * float(K)))
		required_count = int(max(1, min(required_count, K)))

		# Probability that the M-component random KL prior contains at least one
		# positive-support component under the empirical support mass estimate.
		M_prior = int(max(1, self.n_ensemble))
		coverage = 1.0 - torch.pow((1.0 - pos_frac).clamp(0.0, 1.0), M_prior)
		coverage_threshold = float(self.tea_sm_coverage_threshold)
		coverage_ok = torch.ones_like(pos_frac, dtype=torch.bool)
		if coverage_threshold > 0.0:
			coverage_ok = (coverage >= coverage_threshold)

		w = (
			(pos_count >= float(required_count))
			& (pos_mean_delta > float(self.tea_sm_mean_margin))
			& coverage_ok
		).float()
		delta = pos_mean_delta - float(self.tea_sm_mean_margin)
		q_teach_summary = q_pi_eval + pos_mean_delta
		info = {
			"sm_pos_count": pos_count.detach(),
			"sm_pos_frac": pos_frac.detach(),
			"sm_coverage": coverage.detach(),
			"sm_max_delta": max_delta.detach(),
			"sm_mean_delta": mean_delta.detach(),
			"sm_pos_mean_delta": pos_mean_delta.detach(),
			"sm_required_count": float(required_count),
		}
		return w.detach(), delta.detach(), q_teach_summary.detach(), q_pi_eval.detach(), info


	@torch.no_grad()
	def _compute_cstr_gate(self, state, goal, a_pi, K, kl_info=None, sub_cand=None):
		"""Certified / calibrated support-transfer routing gate.

		Modules in the current method definition:
		  1. Support certificate: fixed-rho top-tail score U_rho and positive mass.
		  2. Coverability: optional M-prior coverage diagnostic / hard threshold.
		  3. Local density-field certificate: responsibility-weighted local KL field.
		  4. Critic calibration: advantage-LCB / clipped advantage / target gate critic
		     and optional TD-residual penalty.

		The local field certificate checks the actual random covering prior used by
		the reverse-KL estimator, instead of a global teacher mean.  It therefore
		addresses the negative-support failure mode exposed by CSTR-SQ.
		"""
		B = state.size(0)
		K = int(max(1, int(K)))
		rho = float(min(1.0, max(1e-8, self.tea_cstr_frac)))
		m = int(np.ceil(rho * float(K)))
		m = int(max(1, min(m, K)))

		# -------- 1) Screening support certificate --------
		if sub_cand is None:
			sub_cand = self.sample_subgoal_n(state, goal, K)  # [B,K,state_dim]
		else:
			K = int(sub_cand.size(1))
		state_rep = state.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
		sub_flat = sub_cand.reshape(B * K, self.state_dim)
		if self.tea_cstr_eval_mode == "mean":
			dist = self.actor_target(state_rep, sub_flat)
			a_flat = torch.tanh(dist.loc)
		else:
			a_flat, _, _ = self.actor_target.sample(state_rep, sub_flat)
		a_cand = a_flat.reshape(B, K, self.action_dim)

		delta_all, q_cand_summary, q_pi_eval, adv_std = self._candidate_lower_advantages(
			state, goal, a_pi.detach(), a_cand, use_gate_critic=True
		)  # [B,K], [B,K], [B,1], [B,K]

		top_delta, _ = torch.topk(delta_all, k=m, dim=1, largest=True, sorted=True)  # [B,m]
		toprho_mean = top_delta.mean(dim=1, keepdim=True)
		toprho_min = top_delta.min(dim=1, keepdim=True)[0]
		toprho_max = top_delta.max(dim=1, keepdim=True)[0]

		pos_margin = float(self.tea_cstr_pos_margin)
		positive = (delta_all > pos_margin)
		pos_count = positive.float().sum(dim=1, keepdim=True)
		pos_frac = pos_count / float(K)
		if kl_info is not None and isinstance(kl_info, dict) and ("prior_pool_size" in kl_info):
			M_prior = int(max(1, kl_info.get("prior_pool_size", self.n_ensemble)))
		else:
			M_prior = int(max(1, self.n_ensemble))
		coverage = 1.0 - torch.pow((1.0 - pos_frac).clamp(0.0, 1.0), M_prior)

		toprho_ok = (toprho_mean > float(self.tea_cstr_margin))
		if bool(self.tea_cstr_require_pos_count):
			pos_count_ok = (pos_count >= float(m))
		else:
			pos_count_ok = torch.ones_like(pos_count, dtype=torch.bool)
		coverage_ok = torch.ones_like(pos_frac, dtype=torch.bool)
		if float(self.tea_cstr_coverage) > 0.0:
			coverage_ok = (coverage >= float(self.tea_cstr_coverage))

		# -------- 2) Optional RA diagnostic/veto retained for ablations --------
		ra_ok = torch.ones_like(pos_frac, dtype=torch.bool)
		ra_delta = None
		ra_info = None
		if bool(self.tea_cstr_use_ra_veto):
			_, ra_delta, _, _, ra_info = self._compute_ra_cgr_gate(state, goal, a_pi, kl_info)
			ra_ok = (ra_delta > -float(self.tea_cstr_ra_margin))

		# -------- 3) Local density-field certificate --------
		field_ok = torch.ones_like(pos_frac, dtype=torch.bool)
		field_delta_loc = torch.zeros_like(pos_frac)
		field_score = torch.zeros_like(pos_frac)
		field_prior_delta_mean = torch.zeros_like(pos_frac)
		field_prior_delta_min = torch.zeros_like(pos_frac)
		field_prior_delta_max = torch.zeros_like(pos_frac)
		field_resp_entropy = torch.zeros_like(pos_frac)
		field_eff_components = torch.ones_like(pos_frac)
		field_pos_resp_mass = torch.zeros_like(pos_frac)
		if bool(self.tea_cstr_use_field_veto):
			if kl_info is None or kl_info.get("prior_mode", None) != "all":
				raise RuntimeError("CSTR field veto requires KL info from the random covering prior; use tea_prior_mode=all.")
			for req_key in ["responsibilities", "prior_loc", "prior_scale"]:
				if req_key not in kl_info:
					raise RuntimeError(f"CSTR field veto missing kl_info['{req_key}'].")
			resp = kl_info["responsibilities"].detach()  # [B,M]
			prior_loc = kl_info["prior_loc"].detach()      # [B,M,A]
			prior_scale = kl_info["prior_scale"].detach()  # [B,M,A]
			M_field = int(resp.size(1))
			if self.tea_cstr_field_eval_mode == "mean":
				a_prior = torch.tanh(prior_loc)
			else:
				a_prior = torch.tanh(D.Normal(prior_loc, prior_scale).sample())
			prior_delta, _, _, _ = self._candidate_lower_advantages(
				state, goal, a_pi.detach(), a_prior, use_gate_critic=True
			)  # [B,M]
			field_delta_loc = (resp * prior_delta).sum(dim=1, keepdim=True)
			field_prior_delta_mean = prior_delta.mean(dim=1, keepdim=True)
			field_prior_delta_min = prior_delta.min(dim=1, keepdim=True)[0]
			field_prior_delta_max = prior_delta.max(dim=1, keepdim=True)[0]
			field_resp_entropy = -(resp * torch.log(resp + self.epsilon)).sum(dim=1, keepdim=True)
			field_eff_components = torch.exp(field_resp_entropy)
			field_pos_resp_mass = (resp * (prior_delta > pos_margin).float()).sum(dim=1, keepdim=True)
			if self.tea_cstr_field_mode == "strict":
				field_score = field_delta_loc
			else:
				field_score = field_delta_loc + float(self.tea_cstr_field_lambda) * toprho_mean
			field_ok = (field_score > float(self.tea_cstr_field_margin))

		# -------- 4) Legacy CSTR-SQ global-mean guard (diagnostic/ablation only) --------
		mean_delta = delta_all.mean(dim=1, keepdim=True)
		mean_delta_ok = (mean_delta > float(self.tea_cstr_mean_delta_min))

		w = (toprho_ok & pos_count_ok & coverage_ok & ra_ok & field_ok & mean_delta_ok).float()
		delta = toprho_mean - float(self.tea_cstr_margin)
		q_teach_summary = q_pi_eval + toprho_mean
		max_delta = delta_all.max(dim=1, keepdim=True)[0]
		neg_mask = (delta_all <= pos_margin).float()
		neg_count = neg_mask.sum(dim=1, keepdim=True)
		neg_mean_delta = (delta_all * neg_mask).sum(dim=1, keepdim=True) / (neg_count + self.epsilon)
		neg_mean_delta = torch.where(neg_count > 0.0, neg_mean_delta, torch.zeros_like(neg_mean_delta))

		info = {
			"cstr_toprho_mean_delta": toprho_mean.detach(),
			"cstr_toprho_min_delta": toprho_min.detach(),
			"cstr_toprho_max_delta": toprho_max.detach(),
			"cstr_toprho_m": float(m),
			"cstr_frac": float(rho),
			"cstr_pos_count": pos_count.detach(),
			"cstr_pos_frac": pos_frac.detach(),
			"cstr_coverage": coverage.detach(),
			"cstr_mean_delta": mean_delta.detach(),
			"cstr_mean_delta_min": float(self.tea_cstr_mean_delta_min),
			"cstr_gate_mean_delta_ok": mean_delta_ok.float().detach(),
			"cstr_max_delta": max_delta.detach(),
			"cstr_neg_mean_delta": neg_mean_delta.detach(),
			"cstr_q_policy": q_pi_eval.detach(),
			"cstr_q_teacher_mean": q_cand_summary.mean(dim=1, keepdim=True).detach(),
			"cstr_adv_std_mean": adv_std.mean(dim=1, keepdim=True).detach(),
			"cstr_calib_penalty": float(self._calibration_penalty_value()),
			"cstr_calib_td_last": float(getattr(self, "_calib_td_last", 0.0)),
			"cstr_calib_td_ema": float(getattr(self, "_calib_td_ema", 0.0)),
			"cstr_gate_critic_target": float(1.0 if self.tea_gate_critic == "target" else 0.0),
			"cstr_gate_adv_lcb": float(1.0 if self.tea_calib_adv_mode == "adv_lcb" else 0.0),
			"cstr_gate_toprho_ok": toprho_ok.float().detach(),
			"cstr_gate_pos_count_ok": pos_count_ok.float().detach(),
			"cstr_gate_coverage_ok": coverage_ok.float().detach(),
			"cstr_gate_ra_ok": ra_ok.float().detach(),
			"cstr_gate_field_ok": field_ok.float().detach(),
			"cstr_field_delta_loc": field_delta_loc.detach(),
			"cstr_field_score": field_score.detach(),
			"cstr_field_lambda": float(self.tea_cstr_field_lambda),
			"cstr_field_margin": float(self.tea_cstr_field_margin),
			"cstr_field_prior_delta_mean": field_prior_delta_mean.detach(),
			"cstr_field_prior_delta_min": field_prior_delta_min.detach(),
			"cstr_field_prior_delta_max": field_prior_delta_max.detach(),
			"cstr_field_resp_entropy": field_resp_entropy.detach(),
			"cstr_field_eff_components": field_eff_components.detach(),
			"cstr_field_pos_resp_mass": field_pos_resp_mass.detach(),
		}
		if ra_delta is not None:
			info["cstr_ra_delta"] = ra_delta.detach()
			if ra_info is not None:
				for k_ra, v_ra in ra_info.items():
					info[f"cstr_{k_ra}"] = v_ra.detach() if torch.is_tensor(v_ra) else v_ra
		return w.detach(), delta.detach(), q_teach_summary.detach(), q_pi_eval.detach(), info

	def _tea_log_gate_info(self, info, t: int, tag: str):
		"""TensorBoard logging for extra gate diagnostics."""
		if self.writer is None or info is None:
			return
		try:
			for key, value in info.items():
				if torch.is_tensor(value):
					self.writer.add_scalar(f"{tag}/{key}", float(value.float().mean().item()), int(t))
				else:
					self.writer.add_scalar(f"{tag}/{key}", float(value), int(t))
		except Exception:
			return

	def train_highlevel_policy(self, state, goal, subgoal, t, idx2=None):
		# Compute subgoal distribution
		subgoal_distribution = self.subgoal_net(state, goal)

		with torch.no_grad():
			# Compute target value
			new_subgoal = subgoal_distribution.loc
			policy_v_1 = self.value(state, new_subgoal, idx2=idx2)
			policy_v_2 = self.value(new_subgoal, goal, idx2=idx2)
			policy_v = torch.cat([policy_v_1, policy_v_2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]

			# Compute subgoal distance loss
			v_1 = self.value(state, subgoal, idx2=idx2)
			v_2 = self.value(subgoal, goal, idx2=idx2)
			v = torch.cat([v_1, v_2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]
			adv = -(v - policy_v)
			weight = F.softmax(adv / self.Lambda, dim=0)

		log_prob = subgoal_distribution.log_prob(subgoal).sum(-1)
		subgoal_loss = -(log_prob * weight).mean()

		# Update network
		self.subgoal_optimizer.zero_grad()
		subgoal_loss.backward()
		self.subgoal_optimizer.step()

		# Log variables
		if self.logger is not None:
			self.logger.store(
				adv=adv.mean().item(),
				ratio_adv=adv.ge(0.0).float().mean().item(),
			)
		if self.writer is not None:
			self.writer.add_scalar("hl_adv", adv.mean().item(), t)
			self.writer.add_scalar("hl_ratio_adv", adv.ge(0.0).float().mean().item(), t)
			self.writer.add_scalar("hl_subgoal_loss_mle", subgoal_loss.item(), t)
			self.writer.add_scalar("v_s_subgoal", policy_v_1[0].item(), t)
			self.writer.add_scalar("v_subgoal_goal", policy_v_2[0].item(), t)

	def _tea_gate(self, delta):
		"""Compute per-sample gate w in [0,1] from delta.

		delta: [B,1]
		"""
		if self.tea_hard_gate:
			w = (delta > self.tea_margin).float()
		else:
			# numeric stability: avoid too small temp
			temp = max(self.tea_temp, 1e-6)
			w = torch.sigmoid((delta - self.tea_margin) / temp)
		return w.detach()

	def _tea_gate_choose_plus_from_stats(self, mu_teacher, std_teacher, mu_pi, std_pi):
		"""Choose+ gate: conservative hard routing based on standardized teacher-effect.

		We interpret ensemble Q estimates as Gaussian / sub-Gaussian and compute
			z = (mu_teacher - mu_pi) / sqrt(std_teacher^2 + std_pi^2 + eps).
		Then we imitate iff z >= k_high (confidence threshold in "sigma" units).

		Returns:
			w_hard: [B,1] in {0,1}, detached
			z:      [B,1] standardized effect, detached (useful for calibration plots)
			gap:    [B,1] conservative gap LCB(Q_teacher) - UCB(Q_pi), detached
		"""
		eps = float(self.tea_chooseplus_eps)
		den = torch.sqrt(std_teacher.pow(2) + std_pi.pow(2) + eps)
		z = (mu_teacher - mu_pi) / den
		k_high = float(self.tea_chooseplus_k_high)
		w = (z >= k_high).float()
		# Also expose a conservative gap (useful for analysis / ablations)
		beta = float(self.tea_q_beta)
		gap = (mu_teacher - beta * std_teacher) - (mu_pi + beta * std_pi)
		return w.detach(), z.detach(), gap.detach()

	def _tea_log_delta_bins(self, delta, w, t: int, tag: str = "tea"):
		"""Quantile-binned calibration diagnostics for delta -> gate.

		Logs (per bin):
			- mean(delta), mean(w), fraction of samples
		Also logs histograms for delta and w when enabled.
		"""
		if self.writer is None:
			return
		try:
			if (not self.tea_diag_log_hist) and (not self.tea_diag_bins):
				return
			with torch.no_grad():
				d = delta.detach().view(-1).float()
				ww = w.detach().view(-1).float()
				# Histograms
				if bool(self.tea_diag_log_hist):
					self.writer.add_histogram(f"{tag}/delta_hist", d, int(t))
					self.writer.add_histogram(f"{tag}/gate_hist", ww, int(t))
				# Quantile bins
				B = int(d.numel())
				nb = int(max(0, self.tea_diag_bins))
				if nb <= 0 or B < max(16, nb * 2):
					return
				qs = torch.linspace(0.0, 1.0, nb + 1, device=d.device)
				edges = torch.quantile(d, qs).detach()
				# Ensure strictly increasing edges (avoid empty bins due to ties)
				edges[0] = edges[0] - 1e-6
				edges[-1] = edges[-1] + 1e-6
				for i in range(nb):
					lo = edges[i]
					hi = edges[i + 1]
					mask = (d >= lo) & (d < hi)
					cnt = mask.float().sum()
					if cnt.item() < 1.0:
						continue
					frac = cnt / float(B)
					dm = d[mask].mean()
					wm = ww[mask].mean()
					self.writer.add_scalar(f"{tag}/calib/bin{i:02d}_delta_mean", float(dm.item()), int(t))
					self.writer.add_scalar(f"{tag}/calib/bin{i:02d}_gate_mean", float(wm.item()), int(t))
					self.writer.add_scalar(f"{tag}/calib/bin{i:02d}_frac", float(frac.item()), int(t))
		except Exception:
			# Never break training due to diagnostics.
			return

	def _tea_log_grad_conflict(self, Q_pi, D_KL, w, t: int, tag: str = "tea"):
		"""Gradient conflict diagnostics between RL and imitation terms.

		Computes gradients of:
			L_RL = (-Q_pi).mean()
			L_IM = (alpha * D_KL).mean()
		and logs cosine similarity and norms. This is only called periodically.
		"""
		if self.writer is None or (not bool(self.tea_diag_log_grad)):
			return
		try:
			# Define the two losses (no gating)
			l_rl = (-Q_pi).mean()
			l_im = (self.alpha * D_KL).mean()

			params = [p for p in self.actor.parameters() if p.requires_grad]
			if len(params) == 0:
				return

			g_rl = torch.autograd.grad(l_rl, params, retain_graph=True, create_graph=False, allow_unused=True)
			g_im = torch.autograd.grad(l_im, params, retain_graph=True, create_graph=False, allow_unused=True)

			def _flat(gs):
				vec = []
				for g in gs:
					if g is None:
						continue
					vec.append(g.reshape(-1))
				if len(vec) == 0:
					return None
				return torch.cat(vec, dim=0)

			v_rl = _flat(g_rl)
			v_im = _flat(g_im)
			if (v_rl is None) or (v_im is None):
				return
			eps = 1e-12
			n_rl = torch.norm(v_rl) + eps
			n_im = torch.norm(v_im) + eps
			cos = torch.dot(v_rl, v_im) / (n_rl * n_im)
			conflict = (cos < 0).float()

			# Cancellation ratio for the *mixed* gradient direction (for context only)
			w_bar = float(w.detach().mean().item())
			v_mix = v_rl + (w_bar * v_im)  # alpha absorbed into v_im definition above
			cancel = torch.norm(v_mix) / (n_rl + w_bar * n_im + eps)

			self.writer.add_scalar(f"{tag}/grad_cos", float(cos.item()), int(t))
			self.writer.add_scalar(f"{tag}/grad_conflict", float(conflict.item()), int(t))
			self.writer.add_scalar(f"{tag}/grad_norm_rl", float(n_rl.item()), int(t))
			self.writer.add_scalar(f"{tag}/grad_norm_im", float(n_im.item()), int(t))
			self.writer.add_scalar(f"{tag}/grad_cancel_ratio", float(cancel.item()), int(t))
		except Exception:
			return


	def _spred_weight_ensqfilter(self, q_teacher_set, q_pi_set):
		"""Ensemble Q-filter weight (binary).

		Mirrors the classic Q-filter / ensemble-Q-filter gating:
			w = I[ E[Q(s,a_teacher,g)] >= E[Q(s,a_pi,g)] ]

		Args:
			q_teacher_set: [B, n_Q]
			q_pi_set:      [B, n_Q]
		Returns:
			w: [B,1] in {0,1}
		"""
		mu_teacher = q_teacher_set.mean(-1, keepdim=True)
		mu_pi = q_pi_set.mean(-1, keepdim=True)
		# NOTE: Do not detach here; the caller controls whether w carries gradient.
		return (mu_teacher >= mu_pi).float()


	def _spred_weight_p(self, q_teacher_set, q_pi_set):
		"""SPReD-P (parametric) weight.

		Interprets ensemble Q-values as Gaussian and computes the probability
		that the teacher action improves over the policy action.

		Reference semantics (SPReD):
			z = (mu_teacher - mu_pi) / sqrt(std_teacher^2 + std_pi^2)
			w = 0.5 * (1 + erf(z / sqrt(2)))

		Args:
			q_teacher_set: [B, n_Q]
			q_pi_set:      [B, n_Q]
		Returns:
			w: [B,1] in [0,1]
		"""
		mu_teacher = q_teacher_set.mean(-1, keepdim=True)
		mu_pi = q_pi_set.mean(-1, keepdim=True)
		diff = (mu_teacher - mu_pi)
		std_teacher = q_teacher_set.std(-1, keepdim=True)  # PyTorch default (unbiased=True)
		std_pi = q_pi_set.std(-1, keepdim=True)
		if self.spred_strict:
			denom2 = std_teacher.pow(2) + std_pi.pow(2)
			denom = torch.sqrt(denom2)
			# Handle denom==0 in a limit-consistent way to avoid 0/0 -> NaN.
			mask0 = (denom == 0)
			z = diff / denom
			posinf = torch.tensor(float('inf'), device=z.device, dtype=z.dtype)
			neginf = torch.tensor(float('-inf'), device=z.device, dtype=z.dtype)
			z = torch.where(mask0 & (diff > 0), posinf, z)
			z = torch.where(mask0 & (diff < 0), neginf, z)
			z = torch.where(mask0 & (diff == 0), torch.zeros_like(z), z)
		else:
			eps = 1e-12
			denom = torch.sqrt(std_teacher.pow(2) + std_pi.pow(2) + eps)
			z = diff / denom
			sqrt2 = torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))
			w = 0.5 * (1.0 + torch.erf(z / sqrt2))
			w = torch.nan_to_num(w, nan=0.5, posinf=1.0, neginf=0.0)
			return w.clamp(0.0, 1.0)
		sqrt2 = torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))
		w = 0.5 * (1.0 + torch.erf(z / sqrt2))
		return w.clamp(0.0, 1.0)


	def _spred_weight_e(self, q_teacher_set, q_pi_set):
		"""SPReD-E (robust exponential) weight.

		Uses the interquartile range (IQR) as a robust scale estimator and
		maps advantage into [0,1] via an exponential transform.

		Reference semantics (SPReD):
			beta = (IQR_teacher + IQR_pi)/2 * 10
			w = clamp(exp((mu_teacher - mu_pi)/beta) - 1, 0, 1)

		Args:
			q_teacher_set: [B, n_Q]
			q_pi_set:      [B, n_Q]
		Returns:
			w: [B,1] in [0,1]
		"""
		mu_teacher = q_teacher_set.mean(-1, keepdim=True)
		mu_pi = q_pi_set.mean(-1, keepdim=True)
		diff = (mu_teacher - mu_pi)
		q1_teacher = torch.quantile(q_teacher_set, 0.25, dim=-1, keepdim=True)
		q3_teacher = torch.quantile(q_teacher_set, 0.75, dim=-1, keepdim=True)
		q1_pi = torch.quantile(q_pi_set, 0.25, dim=-1, keepdim=True)
		q3_pi = torch.quantile(q_pi_set, 0.75, dim=-1, keepdim=True)
		iqr_teacher = q3_teacher - q1_teacher
		iqr_pi = q3_pi - q1_pi
		beta = (iqr_teacher + iqr_pi) / 2.0 * 10.0
		if self.spred_strict:
			zero = (beta == 0)
			ratio = diff / beta
			# Handle beta==0 by limit of the exp transform.
			posinf = torch.tensor(float('inf'), device=ratio.device, dtype=ratio.dtype)
			neginf = torch.tensor(float('-inf'), device=ratio.device, dtype=ratio.dtype)
			ratio = torch.where(zero & (diff > 0), posinf, ratio)
			ratio = torch.where(zero & (diff < 0), neginf, ratio)
			ratio = torch.where(zero & (diff == 0), torch.zeros_like(ratio), ratio)
			w = torch.exp(ratio) - 1.0
			# torch.exp(+inf)->inf; clamp to [0,1] matches SPReD.
			w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0)
			return w.clamp(0.0, 1.0)
		else:
			eps = 1e-12
			beta = beta.clamp(min=eps)
			ratio = diff / beta
			ratio = torch.nan_to_num(ratio, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
			w = torch.exp(ratio) - 1.0
			w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0)
			return w.clamp(0.0, 1.0)


	def train(self, state, action, reward, next_state, done, goal, subgoal, t):
			# Dispatch by tea_mode:
			# - 'gated_kl' / 'choose': keep the original RIS/TEA update (n_Q=2) unchanged.
			# - 'ensqfilter' / 'spredp' / 'sprede': use SPReD-fidelity critic/actor update isolated to these modes.
			if self.tea_mode in ["ensqfilter", "spredp", "sprede"]:
				return self._train_spred(state, action, reward, next_state, done, goal, subgoal, t)
			else:
				return self._train_ris_base(state, action, reward, next_state, done, goal, subgoal, t)

	def _train_ris_base(self, state, action, reward, next_state, done, goal, subgoal, t):
			"""Main training step."""
			# Encode images (if vision-based environment), use data augmentation
			if self.image_env:
				state = state.view(-1, 3, 84, 84)
				next_state = next_state.view(-1, 3, 84, 84)
				goal = goal.view(-1, 3, 84, 84)
				subgoal = subgoal.view(-1, 3, 84, 84)

				# Data augmentation
				state = random_translate(state, pad=8)
				next_state = random_translate(next_state, pad=8)
				goal = random_translate(goal, pad=8)
				subgoal = random_translate(subgoal, pad=8)

				# Stop gradient for subgoal/goal and next state
				state = self.encoder(state)
				with torch.no_grad():
					goal = self.encoder(goal)
					next_state = self.encoder(next_state)
					subgoal = self.encoder(subgoal)

			# -------- Critic --------
			with torch.no_grad():
				next_action, _, _ = self.actor.sample(next_state, goal)
				target_Q = self.critic_target(next_state, next_action, goal)
				target_Q = torch.min(target_Q, -1, keepdim=True)[0]
				target_Q = reward + (1.0 - done) * self.gamma * target_Q

			Q = self.critic(state, action, goal)
			critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()
			self._update_td_calibration(Q, target_Q)

			if self.image_env:
				self.encoder_optimizer.zero_grad()
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			if self.image_env:
				self.encoder_optimizer.step()
			self.critic_optimizer.step()

			# Stop backpropagation to encoder
			if self.image_env:
				state = state.detach()
				goal = goal.detach()
				subgoal = subgoal.detach()

			# -------- High-level policy learning --------
			self.train_highlevel_policy(state, goal, subgoal, t)

			# -------- Actor --------
			kl_info = None
			cstr_support_pool = None
			if self.tea_mode == "cstr" and self.tea_cstr_prior_source != "independent":
				cstr_support_pool = self.sample_subgoal_n(state, goal, self.tea_best_of)
				if self.tea_cstr_prior_source == "samek_all":
					prior_subgoal = cstr_support_pool
				else:
					K_pool = int(cstr_support_pool.size(1))
					M_subset = int(min(max(1, self.tea_cstr_prior_subset_m), K_pool))
					idx = torch.randperm(K_pool, device=state.device)[:M_subset]
					prior_subgoal = cstr_support_pool.index_select(1, idx)
				a_pi, D_KL, kl_info = self.sample_action_and_KL(state, goal, return_info=True, subgoal=prior_subgoal)
				if kl_info is not None:
					kl_info["prior_source"] = str(self.tea_cstr_prior_source)
					kl_info["support_pool_size"] = int(cstr_support_pool.size(1))
					kl_info["prior_pool_size"] = int(prior_subgoal.size(1))
			elif self.tea_mode in ["ra_cgr", "cstr"]:
				a_pi, D_KL, kl_info = self.sample_action_and_KL(state, goal, return_info=True)
			else:
				a_pi, D_KL = self.sample_action_and_KL(state, goal)
			Q_pi_all = self.critic(state, a_pi, goal)
			Q_pi = torch.min(Q_pi_all, -1, keepdim=True)[0]

			# TEA gating
			w = None
			delta = None
			q_teach = None
			q_pi_eval = None
			gate_info = None

			if self.tea_enabled and (self.total_it >= self.tea_warmup_steps):
				with torch.no_grad():
					if self.tea_mode == "ra_cgr":
						# Responsibility-Aligned CGR: route using the same random covering prior
						# that produced D_KL, weighted by mixture responsibilities at the actor sample.
						w, delta, q_teach, q_pi_eval, gate_info = self._compute_ra_cgr_gate(state, goal, a_pi, kl_info)
					elif self.tea_mode == "support_mass":
						# Support-Mass CGR: require multiple positive teacher candidates rather
						# than a single best-of-K outlier.
						w, delta, q_teach, q_pi_eval, gate_info = self._compute_support_mass_gate(state, goal, a_pi, self.tea_best_of)
					elif self.tea_mode == "cstr":
						# CSTR: certify top-rho teacher support, M-prior coverage, and optional RA veto.
						w, delta, q_teach, q_pi_eval, gate_info = self._compute_cstr_gate(state, goal, a_pi, self.tea_best_of, kl_info, sub_cand=cstr_support_pool)
					else:
						# Existing CGR/TEA modes: teacher best-of-K by conservative Q.
						a_teach, q_teach = self._sample_teacher_action_bestofk(state, goal, self.tea_best_of)
						if self.tea_mode == "choose_plus":
							# Choose+: confidence-based hard routing using standardized teacher-effect (z-score).
							# Compute ensemble stats for teacher and policy actions under the same (s,g).
							_, mu_t, std_t = self._critic_q_stats(state, a_teach, goal)
							_, mu_pi, std_pi = self._critic_q_stats(state, a_pi, goal)
							w, delta, gap = self._tea_gate_choose_plus_from_stats(mu_t, std_t, mu_pi, std_pi)
							# For consistent logging downstream
							q_teach = mu_t.detach()
							q_pi_eval = mu_pi.detach()
							# Snapshot for analysis (optional)
							self._diag_last["tea_gap"] = float(gap.mean().item())
						else:
							q_pi_eval = self._critic_q_eval(state, a_pi, goal)
							delta = q_teach - q_pi_eval
							w = self._tea_gate(delta)
			else:
				# fallback to original RIS: always use KL
				w = torch.ones_like(Q_pi).detach()

			# Update low-variance TEA diagnostics (EMA) for eval-time correlations
			if (delta is not None) and (w is not None):
				try:
					self._update_diag_ema("delta", float(delta.mean().item()))
					self._update_diag_ema("gate", float(w.mean().item()))
					self._diag_last["delta"] = float(delta.mean().item())
					self._diag_last["gate"] = float(w.mean().item())
				except Exception:
					pass
			# Periodic calibration diagnostics: delta -> gate mapping
			if (delta is not None) and (w is not None) and (self.total_it % self.tea_diag_every == 0):
				self._tea_log_delta_bins(delta, w, t=int(t), tag=f"tea_diag/{self.tea_mode}")
				self._tea_log_gate_info(gate_info, t=int(t), tag=f"tea_diag/{self.tea_mode}")

			# Optional oracle (expert) gate: for analysis / ablation only
			w_oracle, kl_student, kl_teacher = self._oracle_gate_from_expert(state, goal, self.tea_best_of)
			if (w_oracle is not None):
				# Optionally override TEA gate with oracle decision (cheating; use only for ablation)
				if self.tea_oracle_use_for_gating and self.tea_enabled and (self.total_it >= self.tea_warmup_steps):
					w = w_oracle.detach()
				# Agreement diagnostic between TEA gate and oracle gate (thresholded at 0.5)
				try:
					tea_h = (w > 0.5).float()
					agree = (tea_h == w_oracle).float().mean().item()
				except Exception:
					agree = None
			else:
				agree = None
			if self.tea_mode == "gated_kl" or (not self.tea_enabled):
				actor_loss = (self.alpha * w * D_KL - Q_pi).mean()
			else:
				# choose: gate determines whether to do KL-only (w=1) or RL-only (w=0)
				# Ensure hard gate behavior for interpretability
				w_hard = (w > 0.5).float().detach()
				w = w_hard.float()
				eps = float(self.tea_minor_coef)

				if eps > 0.0:
					# KL advantage: w=1 -> KL coef=alpha, Q coef=eps
					# Q  advantage: w=0 -> Q  coef=1,    KL coef=alpha*eps
					kl_coef = self.alpha * (w + (1.0 - w) * eps)
					q_coef  = (1.0 - w) + w * eps
					actor_loss = (kl_coef * D_KL - q_coef * Q_pi).mean()
				else:
					# hard choose (original)
					actor_loss = (w * self.alpha * D_KL - (1.0 - w) * Q_pi).mean()

			# Periodic gradient conflict diagnostics (RL vs imitation)
			if (self.tea_enabled and (w is not None) and (self.total_it % self.tea_diag_every == 0)):
				self._tea_log_grad_conflict(Q_pi, D_KL, w, t=int(t), tag=f"tea_diag/{self.tea_mode}")

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update target networks
			self.total_it += 1
			if self.total_it % self.target_update_interval == 0:
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			# Log variables
			if self.logger is not None:
				self.logger.store(
					actor_loss=actor_loss.item(),
					critic_loss=critic_loss.item(),
					D_KL=D_KL.mean().item(),
					tea_gate=w.mean().item(),
				)
				# Oracle gate diagnostics (expert-only; should be absent in final method)
				if w_oracle is not None:
					self.logger.store(
						oracle_gate=w_oracle.mean().item(),
						oracle_kl_student=kl_student.mean().item(),
						oracle_kl_teacher=kl_teacher.mean().item(),
					)
					if agree is not None:
						self.logger.store(oracle_agree=float(agree))
				if delta is not None:
					self.logger.store(
						tea_delta=delta.mean().item(),
						tea_q_teacher=q_teach.mean().item(),
						tea_q_pi_eval=q_pi_eval.mean().item(),
					)
				if gate_info is not None:
					try:
						gate_scalars = {}
						for _k, _v in gate_info.items():
							if torch.is_tensor(_v):
								gate_scalars[f"tea_{_k}"] = float(_v.float().mean().item())
							else:
								gate_scalars[f"tea_{_k}"] = float(_v)
						self.logger.store(**gate_scalars)
					except Exception:
						pass
			if self.writer is not None:
				self.writer.add_scalar("critic_loss", critic_loss.item(), t)
				self.writer.add_scalar("actor_loss", actor_loss.item(), t)
				self.writer.add_scalar("KL", D_KL.mean().item(), t)
				self.writer.add_scalar("tea_minor_coef", self.tea_minor_coef, t)
				if self.tea_mode == "cstr":
					self.writer.add_scalar("tea_diag/calibration/penalty", float(self._calibration_penalty_value()), t)
					self.writer.add_scalar("tea_diag/calibration/td_residual_last", float(getattr(self, "_calib_td_last", 0.0)), t)
					self.writer.add_scalar("tea_diag/calibration/td_residual_ema", float(getattr(self, "_calib_td_ema", 0.0)), t)

			if self.compact_log_enabled and (self.total_it % max(1, int(self.compact_log_every)) == 0):
				self._compact_log_train(
					t=int(t),
					critic_loss=critic_loss,
					actor_loss=actor_loss,
					D_KL=D_KL,
					Q_pi=Q_pi,
					w=w,
					delta=delta,
					q_teach=q_teach,
					q_pi_eval=q_pi_eval,
					gate_info=gate_info,
					kl_info=kl_info,
				)

	def _train_spred(self, state, action, reward, next_state, done, goal, subgoal, t):
			# SPReD-fidelity training step (isolated to tea_mode in {ensqfilter, spredp, sprede}).
			# Key differences from RIS baseline:
			#   - TD target uses min over a random 2-head subset of target critics (as in SPReD / TD3).
			#   - Actor uses mean over all ensemble heads for Q.
			#   - Teacher regularization uses weighted MSE (behavior cloning) to teacher action (SPReD family).
			#
			# Baseline/gate/choose remain unchanged in _train_ris_base.

			# Encode images (if vision-based environment), use data augmentation
			if self.image_env:
				state = state.view(-1, 3, 84, 84)
				next_state = next_state.view(-1, 3, 84, 84)
				goal = goal.view(-1, 3, 84, 84)
				subgoal = subgoal.view(-1, 3, 84, 84)

				state = random_translate(state, pad=8)
				next_state = random_translate(next_state, pad=8)
				goal = random_translate(goal, pad=8)
				subgoal = random_translate(subgoal, pad=8)

				with torch.no_grad():
					subgoal = subgoal.detach()
					goal = goal.detach()
					next_state = next_state.detach()

				state = self.encoder(state)
				next_state = self.encoder(next_state)
				goal = self.encoder(goal)
				subgoal = self.encoder(subgoal)

			# -------- Critic (SPReD/TD3 target) --------
			with torch.no_grad():
				# Deterministic target action (actor_target mean) + clipped noise (TD3 smoothing)
				_, _, next_mean = self.actor_target.sample(next_state, goal)  # tanh mean
				if getattr(self, "spred_policy_noise", 0.0) > 0:
					noise = torch.randn_like(next_mean) * float(self.spred_policy_noise)
					if getattr(self, "spred_noise_clip", 0.0) > 0:
						noise = noise.clamp(-float(self.spred_noise_clip), float(self.spred_noise_clip))
					next_action = (next_mean + noise).clamp(-1.0, 1.0)
				else:
					next_action = next_mean

				target_Q_set = self.critic_target(next_state, next_action, goal)  # [B, n_Q]
				# Random 2-head min (SPReD source)
				if self.critic_n_q >= 2:
					idx2 = torch.randperm(self.critic_n_q, device=target_Q_set.device)[:2]
					target_Q = torch.min(target_Q_set[:, idx2], dim=-1, keepdim=True)[0]
				else:
					target_Q = target_Q_set.mean(dim=-1, keepdim=True)
				target_Q = reward + (1.0 - done) * self.gamma * target_Q

			current_Q_set = self.critic(state, action, goal)  # [B, n_Q]
			target_expand = target_Q.expand_as(current_Q_set)
			# Match SPReD/TD3 ensemble critic regression: MSE over all heads.
			critic_loss = F.mse_loss(current_Q_set, target_expand)

			if self.image_env:
				self.encoder_optimizer.zero_grad()
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			if self.grad_clip_norm and self.grad_clip_norm > 0:
				torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
				if self.image_env:
					torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip_norm)
			if self.image_env:
				self.encoder_optimizer.step()
			self.critic_optimizer.step()

			# Stop backpropagation to encoder
			if self.image_env:
				state = state.detach()
				goal = goal.detach()
				subgoal = subgoal.detach()

			# -------- High-level policy learning (keep identical to RIS) --------
			self.train_highlevel_policy(state, goal, subgoal, t)

			# -------- Actor (delayed updates, SPReD style) --------
			self.total_it += 1
			policy_freq = int(getattr(self, "spred_policy_freq", 2))
			if policy_freq < 1:
				policy_freq = 1

			actor_loss = None
			bc_loss = None
			w_mean = None

			if (self.total_it % policy_freq) == 0:
				# Policy actions: deterministic mean (TD3/SPReD)
				_, _, a_pi = self.actor.sample(state, goal)  # [B, action_dim]
				q_pi_set = self.critic(state, a_pi, goal)    # [B, n_Q]
				q_pi = q_pi_set.mean(dim=-1, keepdim=True)   # [B, 1]

				# Teacher action: best-of-K, deterministic mean actions from actor_target
				with torch.no_grad():
					B = state.size(0)
					subgoals = self.sample_subgoal(state, goal)  # [B, n_ensemble, state_dim]
					K = int(min(self.tea_best_of, subgoals.size(1)))
					subgoals = subgoals[:, :K, :]
					state_rep = state.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
					sub_flat = subgoals.reshape(B * K, self.state_dim)
					_, _, a_flat_mean = self.actor_target.sample(state_rep, sub_flat)  # mean tanh
					a_cand = a_flat_mean.reshape(B, K, self.action_dim)

					goal_rep = goal.unsqueeze(1).expand(B, K, self.state_dim).reshape(B * K, self.state_dim)
					q_cand = self.critic(state_rep, a_flat_mean, goal_rep).mean(dim=-1, keepdim=True).reshape(B, K, 1)
					q_best, idx = q_cand.max(dim=1)
					idx = idx.view(B, 1, 1).expand(B, 1, self.action_dim)
					a_teach = a_cand.gather(dim=1, index=idx).squeeze(1)  # [B, action_dim]

				# Teacher-regularization term (SPReD variants)
				if self.tea_enabled:
					q_teach_set = self.critic(state, a_teach, goal)  # [B, n_Q]
					se = (a_pi - a_teach).pow(2).mean(dim=-1)  # [B]

					if self.tea_mode == "ensqfilter":
						# EnsQ-filter (source): binary accept if Q_demo >= Q_policy.
						q_teach = q_teach_set.mean(dim=-1)  # [B]
						q_pol = q_pi_set.mean(dim=-1)       # [B]
						mask = torch.ge(q_teach, q_pol).reshape(B, 1).repeat(1, self.action_dim)  # [B, action_dim]
						# Use masked_select to match the source implementation exactly.
						if mask.any():
							bc_loss = F.mse_loss(torch.masked_select(a_pi, mask), torch.masked_select(a_teach, mask))
						else:
							bc_loss = torch.zeros((), device=state.device)
						w_mean = float(mask.sum(dim=0)[0].detach().cpu().item() / max(1, B))
					elif self.tea_mode == "spredp":
						q_pol_mean = q_pi_set.mean(dim=-1)
						q_teach_mean = q_teach_set.mean(dim=-1)
						q_pol_std = torch.std(q_pi_set, dim=-1)
						q_teach_std = torch.std(q_teach_set, dim=-1)
						den = torch.sqrt(q_teach_std**2 + q_pol_std**2 + 1e-12)
						z_score = (q_teach_mean - q_pol_mean) / den
						prob_w = 0.5 * (1.0 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0, device=z_score.device))))
						if self.spred_detach_w:
							prob_w = prob_w.detach()
						bc_loss = (se * prob_w).mean()
						w_mean = prob_w.mean().item()
					else:
						q_pol_mean = q_pi_set.mean(dim=-1)
						q_teach_mean = q_teach_set.mean(dim=-1)
						q_pol_q75 = torch.quantile(q_pi_set, 0.75, dim=-1)
						q_pol_q25 = torch.quantile(q_pi_set, 0.25, dim=-1)
						q_teach_q75 = torch.quantile(q_teach_set, 0.75, dim=-1)
						q_teach_q25 = torch.quantile(q_teach_set, 0.25, dim=-1)
						q_pol_iqr = q_pol_q75 - q_pol_q25
						q_teach_iqr = q_teach_q75 - q_teach_q25
						beta = (q_pol_iqr + q_teach_iqr) / 2.0 * 10.0
						beta = beta.clamp(min=1e-12)
						exp_w = torch.exp((q_teach_mean - q_pol_mean) / beta) - 1.0
						exp_w = torch.clamp(exp_w, min=0.0, max=1.0)
						if self.spred_detach_w:
							exp_w = exp_w.detach()
						bc_loss = (se * exp_w).mean()
						w_mean = exp_w.mean().item()
				else:
					bc_loss = torch.zeros((), device=state.device)


				# SPReD source: actor_loss = -lambda1 * Q.mean() + lambda2 * BC_loss
				actor_loss = (-self.spred_lambda1 * q_pi.mean()) + (self.spred_lambda2 * bc_loss)
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				if self.grad_clip_norm and self.grad_clip_norm > 0:
					torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
				self.actor_optimizer.step()

				# Update target networks (SPReD source: update after delayed policy step)
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			# Logging (lightweight; diagnostics handled elsewhere)
			if self.writer is not None:
				self.writer.add_scalar("critic_loss", critic_loss.item(), t)
				if actor_loss is not None:
					self.writer.add_scalar("actor_loss", float(actor_loss.item()), t)
				if bc_loss is not None:
					self.writer.add_scalar("spred_bc_loss", float(bc_loss.item()), t)
				if w_mean is not None:
					self.writer.add_scalar("spred_w_mean", float(w_mean), t)

	def _update_diag_ema(self, name: str, value: float):
		"""Exponential moving average for low-variance diagnostics."""
		beta = float(self.tea_diag_ema_beta)
		v = float(value)
		if name not in self._diag_ema:
			self._diag_ema[name] = v
		else:
			self._diag_ema[name] = beta * float(self._diag_ema[name]) + (1.0 - beta) * v

	def record_eval(self, success_rate: float, t: int, tag: str = "random", eval_distance=None):
		"""Record an evaluation point and log correlations with gate / delta diagnostics.

		This is intentionally lightweight and can be called from the training loop
		whenever an eval is executed.
		"""
		# Compact eval rows are useful even when TensorBoard events are skipped.
		try:
			self._compact_log_eval(tag=tag, t=int(t), success_rate=float(success_rate), eval_distance=eval_distance)
		except Exception:
			pass
		if not self.tea_enabled:
			return
		try:
			import numpy as _np
		except Exception:
			return
		delta_ema = float(self._diag_ema.get("delta", _np.nan))
		gate_ema = float(self._diag_ema.get("gate", _np.nan))
		success = float(success_rate)
		snap = {"t": int(t), "success": success, "eval_distance": self._compact_float(eval_distance), "delta_ema": delta_ema, "gate_ema": gate_ema}
		hist = self._eval_hist.setdefault(str(tag), [])
		hist.append(snap)
		# keep a bounded window
		if len(hist) > self.tea_eval_corr_window:
			del hist[0:len(hist) - self.tea_eval_corr_window]

		def _pearson(x, y):
			x = _np.asarray(x, dtype=_np.float64)
			y = _np.asarray(y, dtype=_np.float64)
			mask = _np.isfinite(x) & _np.isfinite(y)
			x = x[mask]
			y = y[mask]
			if x.size < 3:
				return _np.nan
			if _np.std(x) < 1e-12 or _np.std(y) < 1e-12:
				return _np.nan
			return float(_np.corrcoef(x, y)[0, 1])

		# correlations (same-time)
		success_seq = [h["success"] for h in hist]
		delta_seq = [h["delta_ema"] for h in hist]
		gate_seq = [h["gate_ema"] for h in hist]
		corr_delta_0 = _pearson(delta_seq, success_seq)
		corr_gate_0 = _pearson(gate_seq, success_seq)
		# simple lag-1 correlations: x_{i-1} vs y_i
		corr_delta_l1 = _pearson(delta_seq[:-1], success_seq[1:]) if len(hist) >= 4 else _np.nan
		corr_gate_l1 = _pearson(gate_seq[:-1], success_seq[1:]) if len(hist) >= 4 else _np.nan

		# Log to tb/logger
		if self.logger is not None:
			self.logger.store(
				**{
					f"eval_success_{tag}": success,
					f"eval_delta_ema_{tag}": delta_ema,
					f"eval_gate_ema_{tag}": gate_ema,
					f"eval_corr_delta_success_{tag}": corr_delta_0,
					f"eval_corr_gate_success_{tag}": corr_gate_0,
					f"eval_corr_delta_success_lag1_{tag}": corr_delta_l1,
					f"eval_corr_gate_success_lag1_{tag}": corr_gate_l1,
				}
			)
		if self.writer is not None:
			self.writer.add_scalar(f"tea_eval/{tag}/success", success, t)
			self.writer.add_scalar(f"tea_eval/{tag}/delta_ema", delta_ema, t)
			self.writer.add_scalar(f"tea_eval/{tag}/gate_ema", gate_ema, t)
			if _np.isfinite(corr_delta_0):
				self.writer.add_scalar(f"tea_eval/{tag}/corr_delta_success", corr_delta_0, t)
			if _np.isfinite(corr_gate_0):
				self.writer.add_scalar(f"tea_eval/{tag}/corr_gate_success", corr_gate_0, t)
			if _np.isfinite(corr_delta_l1):
				self.writer.add_scalar(f"tea_eval/{tag}/corr_delta_success_lag1", corr_delta_l1, t)
			if _np.isfinite(corr_gate_l1):
				self.writer.add_scalar(f"tea_eval/{tag}/corr_gate_success_lag1", corr_gate_l1, t)