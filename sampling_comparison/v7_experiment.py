"""Sampling V7 public module.

This file re-exports the maintained implementation symbols explicitly to avoid
star-import indirection while preserving import compatibility.
"""

from sampling_comparison.v7_experiment_repaired import BinaryPopulationResult
from sampling_comparison.v7_experiment_repaired import ReplayPlan
from sampling_comparison.v7_experiment_repaired import V7_BUDGET_LEVELS
from sampling_comparison.v7_experiment_repaired import V7_DEFAULT_BASE_SEED
from sampling_comparison.v7_experiment_repaired import V7_DEFAULT_REPETITIONS
from sampling_comparison.v7_experiment_repaired import V7_METHOD_IDS
from sampling_comparison.v7_experiment_repaired import V7_METHOD_ID_ORDER
from sampling_comparison.v7_experiment_repaired import V7_PCA_DIMENSIONS
from sampling_comparison.v7_experiment_repaired import V7_VERSION
from sampling_comparison.v7_experiment_repaired import build_replay_plan
from sampling_comparison.v7_experiment_repaired import build_reduced_vector_export_documents
from sampling_comparison.v7_experiment_repaired import build_v7_method_matrix
from sampling_comparison.v7_experiment_repaired import build_v7_report_html
from sampling_comparison.v7_experiment_repaired import compute_overlap_diagnostics
from sampling_comparison.v7_experiment_repaired import derive_budget_caps
from sampling_comparison.v7_experiment_repaired import embedding_binary_metrics
from sampling_comparison.v7_experiment_repaired import fit_pca8_embeddings
from sampling_comparison.v7_experiment_repaired import idw_binary_population
from sampling_comparison.v7_experiment_repaired import materialize_replay_dataset_view
from sampling_comparison.v7_experiment_repaired import make_synthetic_trace_for_text
from sampling_comparison.v7_experiment_repaired import run_v7_core
from sampling_comparison.v7_experiment_repaired import run_v7_experiment
from sampling_comparison.v7_experiment_repaired import select_diverse_exact_cap
from sampling_comparison.v7_experiment_repaired import select_minhash_exact_cap

__all__ = [
	"BinaryPopulationResult",
	"ReplayPlan",
	"V7_BUDGET_LEVELS",
	"V7_DEFAULT_BASE_SEED",
	"V7_DEFAULT_REPETITIONS",
	"V7_METHOD_IDS",
	"V7_METHOD_ID_ORDER",
	"V7_PCA_DIMENSIONS",
	"V7_VERSION",
	"build_replay_plan",
	"build_reduced_vector_export_documents",
	"build_v7_method_matrix",
	"build_v7_report_html",
	"compute_overlap_diagnostics",
	"derive_budget_caps",
	"embedding_binary_metrics",
	"fit_pca8_embeddings",
	"idw_binary_population",
	"materialize_replay_dataset_view",
	"make_synthetic_trace_for_text",
	"run_v7_core",
	"run_v7_experiment",
	"select_diverse_exact_cap",
	"select_minhash_exact_cap",
]
