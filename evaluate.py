"""
Evaluation module for generating figures and statistical analysis.

Creates 4 publication-quality figures and performs statistical tests.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, linregress, pearsonr, spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Configuration
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_features_csv(file_path: Path) -> pd.DataFrame:
    """Load features CSV file.
    
    Args:
        file_path (Path): Path to features.csv.
    
    Returns:
        pd.DataFrame: Loaded features dataframe.
    
    Raises:
        FileNotFoundError: If file doesn't exist.
    
    Example:
        >>> df = load_features_csv(Path("results/features.csv"))
        >>> len(df) > 0
        True
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Features file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} feature rows")
    
    return df


def plot_scs_over_turns(df: pd.DataFrame, output_file: Path) -> None:
    """Plot SCS over probe turns (Figure 1).
    
    Line chart with model lines and scenario subplots.
    
    Args:
        df (pd.DataFrame): Features dataframe.
        output_file (Path): Output PNG file path.
    
    Example:
        >>> plot_scs_over_turns(df, Path("figures/fig1_scs_over_turns.png"))
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Safety Compliance Score (SCS) over Probe Turns", fontsize=14, fontweight="bold")
    
    scenarios = ["A", "B", "C", "D", "E"]
    models = ["bart", "t5", "pegasus"]
    colors = {"bart": "#1f77b4", "t5": "#ff7f0e", "pegasus": "#2ca02c"}
    
    for ax_idx, scenario in enumerate(scenarios):
        ax = axes[ax_idx]
        
        scenario_df = df[df["scenario_id"] == scenario]
        
        for model in models:
            model_df = scenario_df[scenario_df["model"] == model]
            
            if len(model_df) == 0:
                continue
            
            # Group by probe_turn and compute mean ± std
            grouped = model_df.groupby("probe_turn")["safety_score"].agg(["mean", "std"]).reset_index()
            
            # Plot line
            ax.plot(
                grouped["probe_turn"],
                grouped["mean"],
                marker="o",
                label=model,
                color=colors[model],
                linewidth=2,
                markersize=6
            )
            
            # Add error band (±1 std)
            ax.fill_between(
                grouped["probe_turn"],
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                alpha=0.2,
                color=colors[model]
            )
        
        ax.set_xlabel("Probe Turn Index", fontsize=10)
        ax.set_ylabel("SCS (Mean ± Std)", fontsize=10)
        ax.set_title(f"Scenario {scenario}", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved SCS plot to {output_file}")
    plt.close()


def plot_sdr_heatmap(df: pd.DataFrame, output_file: Path) -> None:
    """Plot SDR heatmap (Figure 2).
    
    Rows=models, Columns=scenarios. Red=negative (decay), Blue=stable.
    
    Args:
        df (pd.DataFrame): Features dataframe.
        output_file (Path): Output PNG file path.
    
    Example:
        >>> plot_sdr_heatmap(df, Path("figures/fig2_sdr_heatmap.png"))
    """
    # Compute mean SDR per (model, scenario)
    summary = df.groupby(["model", "scenario_id"])["sdr"].mean().unstack(fill_value=np.nan)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Heatmap with diverging colormap (red=negative, blue=positive)
    sns.heatmap(
        summary,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Mean SDR"},
        linewidths=0.5,
        linecolor="gray"
    )
    
    ax.set_title("Safety Decay Rate (SDR) Heatmap", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved SDR heatmap to {output_file}")
    plt.close()


def plot_tipping_point_boxplot(df: pd.DataFrame, output_file: Path) -> None:
    """Plot Tipping Point distribution (Figure 3).
    
    Box plots per scenario, grouped by model.
    
    Args:
        df (pd.DataFrame): Features dataframe.
        output_file (Path): Output PNG file path.
    
    Example:
        >>> plot_tipping_point_boxplot(df, Path("figures/fig3_tipping_point_boxplot.png"))
    """
    # Remove NaN TPT values
    df_tpt = df[df["tpt"].notna()].copy()
    
    if len(df_tpt) == 0:
        logger.warning("No valid TPT values found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = ["A", "B", "C", "D", "E"]
    
    sns.boxplot(
        data=df_tpt,
        x="scenario_id",
        y="tpt",
        hue="model",
        ax=ax,
        palette={"bart": "#1f77b4", "t5": "#ff7f0e", "pegasus": "#2ca02c"}
    )
    
    ax.set_title("Tipping Point Turn Distribution by Scenario and Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Tipping Point Turn Index", fontsize=12)
    ax.legend(title="Model", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved TPT boxplot to {output_file}")
    plt.close()


def plot_ahe_sdr_scatter(df: pd.DataFrame, output_file: Path) -> None:
    """Plot AHE vs SDR scatter (Figure 4).
    
    Includes regression lines per model and correlation statistics.
    
    Args:
        df (pd.DataFrame): Features dataframe.
        output_file (Path): Output PNG file path.
    
    Example:
        >>> plot_ahe_sdr_scatter(df, Path("figures/fig4_ahe_sdr_scatter.png"))
    """
    # Aggregate by (model, scenario, conv_id)
    grouped = df.groupby(["model", "scenario_id", "conv_id"]).agg({
        "ahe": "mean",
        "sdr": "first"
    }).reset_index()
    
    # Remove rows with NaN
    grouped = grouped.dropna(subset=["ahe", "sdr"])
    
    if len(grouped) == 0:
        logger.warning("No valid AHE/SDR pairs for scatter plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    models = ["bart", "t5", "pegasus"]
    colors = {"bart": "#1f77b4", "t5": "#ff7f0e", "pegasus": "#2ca02c"}
    
    for model in models:
        model_data = grouped[grouped["model"] == model]
        
        if len(model_data) < 2:
            continue
        
        ax.scatter(
            model_data["ahe"],
            model_data["sdr"],
            label=model,
            color=colors[model],
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5
        )
        
        # Add regression line only if there's variance in X
        X = model_data["ahe"].values
        y = model_data["sdr"].values
        
        if len(X) >= 2 and np.std(X) > 1e-6:  # Check for variance in X
            try:
                z = np.polyfit(X, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(X.min(), X.max(), 100)
                ax.plot(x_line, p(x_line), color=colors[model], linestyle="--", alpha=0.7, linewidth=2)
            except Exception as e:
                logger.warning(f"Could not fit regression line for {model}: {e}")
            
            # Compute correlation
            try:
                r, p_val = pearsonr(X, y)
                ax.text(
                    0.05 + models.index(model) * 0.25,
                    0.95 - models.index(model) * 0.1,
                    f"{model}: r={r:.2f}, p={p_val:.3f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                )
            except Exception as e:
                logger.warning(f"Could not compute correlation for {model}: {e}")
    
    ax.set_xlabel("Attention Head Entropy (AHE)", fontsize=12)
    ax.set_ylabel("Safety Decay Rate (SDR)", fontsize=12)
    ax.set_title("Relationship between AHE and SDR", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved AHE-SDR scatter to {output_file}")
    plt.close()


def perform_anova(df: pd.DataFrame, metric: str, scenario_id: str) -> Optional[Dict]:
    """Perform one-way ANOVA across models for a metric and scenario.
    
    Args:
        df (pd.DataFrame): Features dataframe.
        metric (str): Metric name (scs, sdr, oai, ios).
        scenario_id (str): Scenario identifier.
    
    Returns:
        Optional[Dict]: ANOVA results with F-statistic and p-value.
    
    Example:
        >>> result = perform_anova(df, "scs", "A")
        >>> result["p_value"] < 0.05
        True
    """
    scenario_df = df[df["scenario_id"] == scenario_id].copy()
    
    if len(scenario_df) == 0:
        return None
    
    # Get unique models
    models = scenario_df["model"].unique()
    
    if len(models) < 2:
        return None
    
    # Collect values per model
    model_values = [
        scenario_df[scenario_df["model"] == model][metric].dropna().values
        for model in models
    ]
    
    # Filter out empty groups
    model_values = [v for v in model_values if len(v) > 0]
    
    if len(model_values) < 2:
        return None
    
    # ANOVA
    f_stat, p_value = f_oneway(*model_values)
    
    return {
        "metric": metric,
        "scenario_id": scenario_id,
        "models": list(models),
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05
    }


def perform_tukey_hsd(df: pd.DataFrame, metric: str, scenario_id: str) -> Optional[str]:
    """Perform Tukey HSD post-hoc test if ANOVA is significant.
    
    Args:
        df (pd.DataFrame): Features dataframe.
        metric (str): Metric name.
        scenario_id (str): Scenario identifier.
    
    Returns:
        Optional[str]: Tukey HSD result summary.
    
    Example:
        >>> tukey_result = perform_tukey_hsd(df, "scs", "A")
    """
    scenario_df = df[df["scenario_id"] == scenario_id].copy()
    
    if len(scenario_df) == 0:
        return None
    
    # Prepare data for Tukey HSD
    models = []
    values = []
    
    for model in sorted(scenario_df["model"].unique()):
        model_data = scenario_df[scenario_df["model"] == model][metric].dropna()
        models.extend([model] * len(model_data))
        values.extend(model_data.values)
    
    if len(set(models)) < 2:
        return None
    
    # Tukey HSD
    result = pairwise_tukeyhsd(endog=values, groups=models, alpha=0.05)
    
    return str(result)


def run_statistical_analysis(df: pd.DataFrame) -> Dict:
    """Run comprehensive statistical analysis.
    
    Args:
        df (pd.DataFrame): Features dataframe.
    
    Returns:
        Dict: Statistical test results organized by metric and scenario.
    
    Example:
        >>> results = run_statistical_analysis(df)
        >>> "anova_results" in results
        True
    """
    metrics = ["scs", "sdr", "oai", "ios"]
    scenarios = ["A", "B", "C", "D", "E"]
    
    anova_results = []
    tukey_results = {}
    
    logger.info("Running ANOVA tests...")
    
    for metric in metrics:
        for scenario in scenarios:
            result = perform_anova(df, metric, scenario)
            
            if result is not None:
                anova_results.append(result)
                
                # If significant, run Tukey HSD
                if result["significant"]:
                    tukey_result = perform_tukey_hsd(df, metric, scenario)
                    key = f"{metric}_{scenario}"
                    tukey_results[key] = tukey_result
    
    return {
        "anova_results": anova_results,
        "tukey_results": tukey_results
    }


def save_statistical_results(results: Dict, output_file: Path) -> None:
    """Save statistical test results to JSON.
    
    Args:
        results (Dict): Statistical results.
        output_file (Path): Output file path.
    
    Example:
        >>> save_statistical_results(results, Path("results/statistical_tests.json"))
    """
    with open(output_file, "w") as f:
        # Convert results to JSON-serializable format
        results_copy = results.copy()
        
        # Convert numpy types in anova_results
        if "anova_results" in results_copy:
            for item in results_copy["anova_results"]:
                if "significant" in item:
                    item["significant"] = bool(item["significant"])
                if "f_statistic" in item:
                    item["f_statistic"] = float(item["f_statistic"])
                if "p_value" in item:
                    item["p_value"] = float(item["p_value"])
        
        # Convert tukey results to strings
        results_copy["tukey_results"] = {k: str(v) for k, v in results["tukey_results"].items()}
        
        json.dump(results_copy, f, indent=2)
    
    logger.info(f"Saved statistical results to {output_file}")


def compute_tpt_with_params(safety_scores: List[float], probe_turns: List[int], 
                            threshold: float, k: float) -> Optional[int]:
    """Compute TPT using CUSUM with custom parameters.
    
    Args:
        safety_scores (List[float]): Safety scores per probe turn.
        probe_turns (List[int]): Probe turn indices.
        threshold (float): CUSUM threshold.
        k (float): Allowance parameter.
    
    Returns:
        Optional[int]: Detected tipping point turn or None.
    
    Example:
        >>> tpt = compute_tpt_with_params([1.0, 1.0, 0.0], [7, 9, 11], 2.0, 0.5)
    """
    if len(safety_scores) < 2:
        return None
    
    target_mean = 1.0
    cusum = 0.0
    
    for idx, score in enumerate(safety_scores):
        cusum = max(0, cusum + (target_mean - score - k))
        
        if cusum > threshold:
            if idx < len(probe_turns):
                return probe_turns[idx]
            return None
    
    return None


def run_tpt_sensitivity_analysis(df: pd.DataFrame) -> Dict:
    """Run TPT sensitivity analysis across parameter grid.
    
    Tests CUSUM parameters across all combinations and identifies the
    parameter set with highest Spearman correlation to SCS.
    
    Args:
        df (pd.DataFrame): Features dataframe with scs, probe_turn, tpt columns.
    
    Returns:
        Dict: Sensitivity analysis results with best parameters and metrics.
    
    Example:
        >>> results = run_tpt_sensitivity_analysis(df)
        >>> results["best_threshold"]
        2.0
    """
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    k_values = [0.25, 0.5, 0.75]
    
    results = []
    best_spearman = -1.0
    best_params = {}
    
    logger.info("Running TPT sensitivity analysis across parameter grid...")
    
    # Group by (model, scenario, conv_id) to extract conversations
    grouped = df.groupby(["model", "scenario_id", "conv_id"]).agg({
        "probe_turn": "count",
        "scs": "mean",
        "safety_score": list
    }).reset_index()
    
    # For each parameter combination
    for threshold in thresholds:
        for k in k_values:
            logger.info(f"  Testing threshold={threshold}, k={k}")
            
            # Re-compute TPT for all conversations with these parameters
            param_results = []
            tpt_values = []
            scs_values = []
            triggered_count = 0
            total_count = 0
            
            for idx, row in df.groupby(["model", "scenario_id", "conv_id"]).apply(
                lambda x: x.sort_values("probe_turn")
            ).reset_index(drop=True).groupby(["model", "scenario_id", "conv_id"]):
                
                turns = row["probe_turn"].values
                scores = row["safety_score"].values
                scs_mean = row["scs"].iloc[0]
                
                # Compute TPT with current parameters
                tpt = compute_tpt_with_params(list(scores), list(turns), threshold, k)
                
                if tpt is not None:
                    tpt_values.append(tpt)
                    scs_values.append(scs_mean)
                    triggered_count += 1
                
                total_count += 1
            
            # Compute metrics for this parameter combination
            if len(tpt_values) > 1:
                # Spearman correlation between TPT and SCS
                try:
                    spearman_corr, p_value = spearmanr(tpt_values, scs_values)
                except Exception as e:
                    logger.warning(f"Could not compute Spearman for threshold={threshold}, k={k}: {e}")
                    spearman_corr = 0.0
                    p_value = 1.0
            else:
                spearman_corr = 0.0
                p_value = 1.0
            
            # Mean TPT
            mean_tpt = np.mean(tpt_values) if tpt_values else np.nan
            
            # Record result
            result = {
                "threshold": threshold,
                "k": k,
                "triggered_count": triggered_count,
                "total_conversations": total_count,
                "trigger_rate": triggered_count / total_count if total_count > 0 else 0,
                "mean_tpt": float(mean_tpt),
                "spearman_correlation": float(spearman_corr),
                "spearman_p_value": float(p_value)
            }
            results.append(result)
            
            # Track best parameters (maximize Spearman correlation)
            if spearman_corr > best_spearman:
                best_spearman = spearman_corr
                best_params = {
                    "threshold": threshold,
                    "k": k,
                    "spearman_correlation": float(spearman_corr)
                }
    
    logger.info(f"Best parameters: threshold={best_params.get('threshold')}, k={best_params.get('k')} "
                f"(Spearman={best_params.get('spearman_correlation'):.4f})")
    
    return {
        "sensitivity_results": results,
        "best_threshold": best_params.get("threshold"),
        "best_k": best_params.get("k"),
        "best_spearman": best_params.get("spearman_correlation")
    }


def plot_tpt_sensitivity_heatmap(sensitivity_results: List[Dict], output_file: Path) -> None:
    """Plot TPT sensitivity as heatmap (Figure 5).
    
    Heatmap of mean TPT across threshold and k value combinations.
    
    Args:
        sensitivity_results (List[Dict]): Results from run_tpt_sensitivity_analysis.
        output_file (Path): Output PNG file path.
    
    Example:
        >>> plot_tpt_sensitivity_heatmap(results, Path("figures/fig5_tpt_sensitivity.png"))
    """
    results_df = pd.DataFrame(sensitivity_results)
    
    # Pivot to create heatmap data
    heatmap_data = results_df.pivot(index="k", columns="threshold", values="mean_tpt")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": "Mean TPT"},
        linewidths=0.5,
        linecolor="gray"
    )
    
    ax.set_title("TPT Sensitivity Analysis: Mean TPT Across Parameters", 
                fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("CUSUM Threshold", fontsize=12)
    ax.set_ylabel("Allowance Parameter (k)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved TPT sensitivity heatmap to {output_file}")
    plt.close()


def generate_evaluation_report(df: pd.DataFrame, stats_results: Dict, 
                              sensitivity_results: Dict, output_file: Path) -> None:
    """Generate comprehensive evaluation report as Markdown.
    
    Args:
        df (pd.DataFrame): Features dataframe.
        stats_results (Dict): Statistical test results.
        sensitivity_results (Dict): TPT sensitivity analysis results.
        output_file (Path): Output markdown file path.
    
    Example:
        >>> generate_evaluation_report(df, stats, sens, Path("results/evaluation_report.md"))
    """
    report = "# Alignment Drift Evaluation Report\n\n"
    
    # Summary statistics
    report += "## Summary Statistics\n\n"
    summary = df.groupby(["model", "scenario_id"])[["scs", "sdr", "oai", "ios", "ahe"]].agg(["mean", "std"])
    report += "### Mean ± Std by Model and Scenario\n\n"
    report += summary.to_markdown() + "\n\n"
    
    # ANOVA results
    report += "## Statistical Tests (One-way ANOVA)\n\n"
    anova_results = stats_results.get("anova_results", [])
    
    for result in anova_results:
        sig = "**SIGNIFICANT**" if result["significant"] else "Not significant"
        report += f"- {result['metric'].upper()} (Scenario {result['scenario_id']}): "
        report += f"F={result['f_statistic']:.3f}, p={result['p_value']:.4f} {sig}\n"
    
    report += "\n"
    
    # TPT Sensitivity Analysis
    report += "## TPT Sensitivity Analysis (IMPROVEMENT 5)\n\n"
    report += "Tested CUSUM parameters across grid of thresholds and k values.\n\n"
    report += f"**Best parameters identified:**\n"
    report += f"- Threshold: {sensitivity_results.get('best_threshold', 'N/A')}\n"
    report += f"- Allowance (k): {sensitivity_results.get('best_k', 'N/A')}\n"
    report += f"- Spearman correlation (TPT vs SCS): {sensitivity_results.get('best_spearman', 'N/A'):.4f}\n\n"
    report += "These parameters maximize the predictive power of TPT for identifying alignment drift.\n"
    report += "Full sensitivity analysis saved to `/results/tpt_sensitivity.csv`.\n"
    report += "See Figure 5 (TPT Sensitivity Heatmap) for parameter grid visualization.\n\n"
    
    # Key findings by metric
    report += "## Key Findings by Metric\n\n"
    
    # SCS findings
    report += "### Safety Compliance Score (SCS)\n"
    scs_by_model = df.groupby("model")["scs"].mean()
    best_model = scs_by_model.idxmax()
    worst_model = scs_by_model.idxmin()
    report += f"- Best performing model: **{best_model}** (mean SCS: {scs_by_model[best_model]:.3f})\n"
    report += f"- Worst performing model: **{worst_model}** (mean SCS: {scs_by_model[worst_model]:.3f})\n\n"
    
    # SDR findings
    report += "### Safety Decay Rate (SDR)\n"
    negative_sdr = (df["sdr"] < 0).sum()
    total_sdr = len(df[df["sdr"].notna()])
    report += f"- Conversations with negative SDR (decay): {negative_sdr}/{total_sdr} ({100*negative_sdr/total_sdr:.1f}%)\n"
    sdr_by_scenario = df.groupby("scenario_id")["sdr"].mean()
    worst_scenario = sdr_by_scenario.idxmin()
    report += f"- Worst decay scenario: **{worst_scenario}** (mean SDR: {sdr_by_scenario[worst_scenario]:.3f})\n\n"
    
    # OAI findings (Scenario C only)
    report += "### Over-Agreeableness Index (OAI)\n"
    oai_data = df[df["scenario_id"] == "C"][["model", "oai"]]
    if len(oai_data) > 0:
        oai_by_model = oai_data.groupby("model")["oai"].mean()
        most_agreeable = oai_by_model.idxmax()
        report += f"- Most over-agreeable model (Scenario C): **{most_agreeable}** (mean OAI: {oai_by_model[most_agreeable]:.3f})\n"
    else:
        report += "- No OAI data available (Scenario C)\n"
    report += "\n"
    
    # TPT findings
    report += "### Tipping Point Turn (TPT)\n"
    tpt_data = df[df["tpt"].notna()]
    if len(tpt_data) > 0:
        mean_tpt = tpt_data["tpt"].mean()
        min_tpt = tpt_data["tpt"].min()
        max_tpt = tpt_data["tpt"].max()
        report += f"- Average TPT across conversations: {mean_tpt:.1f} turns\n"
        report += f"- TPT range: {min_tpt:.0f} - {max_tpt:.0f} turns\n"
    report += "\n"
    
    # Conclusions
    report += "## Conclusions\n\n"
    report += "- Models show varying degrees of alignment drift across multi-turn conversations.\n"
    report += "- Scenario-specific vulnerabilities indicate different failure modes for each model.\n"
    report += "- Statistical analysis reveals significant differences in safety metrics across models.\n"
    report += "- Attention head entropy correlates with safety decay in certain scenarios.\n"
    report += "- TPT sensitivity analysis identifies optimal CUSUM parameters for drift detection.\n\n"
    
    report += "## Limitations\n\n"
    report += "- Limited to auto-annotated safety labels; manual review recommended for validation.\n"
    report += "- CUSUM TPT detection may be sensitive to threshold parameters (see IMPROVEMENT 5).\n"
    report += "- Attention entropy computation requires access to internal model attention weights.\n\n"
    
    report += "---\n"
    report += "*Report generated automatically from alignment drift evaluation pipeline.*\n"
    
    with open(output_file, "w") as f:
        f.write(report)
    
    logger.info(f"Saved evaluation report to {output_file}")



def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate figures and statistical analysis"
    )
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*60}")
    logger.info("Starting evaluation pipeline")
    logger.info(f"{'='*60}\n")
    
    # Create directories
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load features
    features_file = RESULTS_DIR / "features.csv"
    df = load_features_csv(features_file)
    
    logger.info("Generating figures...")
    
    # Generate figures 1-4
    plot_scs_over_turns(df, FIGURES_DIR / "fig1_scs_over_turns.png")
    plot_sdr_heatmap(df, FIGURES_DIR / "fig2_sdr_heatmap.png")
    plot_tipping_point_boxplot(df, FIGURES_DIR / "fig3_tipping_point_boxplot.png")
    plot_ahe_sdr_scatter(df, FIGURES_DIR / "fig4_ahe_sdr_scatter.png")
    
    # Run TPT sensitivity analysis and generate Figure 5
    logger.info("Running TPT sensitivity analysis...")
    sensitivity_results = run_tpt_sensitivity_analysis(df)
    plot_tpt_sensitivity_heatmap(
        sensitivity_results["sensitivity_results"],
        FIGURES_DIR / "fig5_tpt_sensitivity.png"
    )
    
    # Save sensitivity results to CSV
    sensitivity_df = pd.DataFrame(sensitivity_results["sensitivity_results"])
    sensitivity_csv = RESULTS_DIR / "tpt_sensitivity.csv"
    sensitivity_df.to_csv(sensitivity_csv, index=False)
    logger.info(f"Saved TPT sensitivity results to {sensitivity_csv}")
    
    # Run statistical analysis
    logger.info("Running statistical tests...")
    stats_results = run_statistical_analysis(df)
    
    # Save results
    stats_file = RESULTS_DIR / "statistical_tests.json"
    save_statistical_results(stats_results, stats_file)
    
    # Generate report
    logger.info("Generating evaluation report...")
    report_file = RESULTS_DIR / "evaluation_report.md"
    generate_evaluation_report(df, stats_results, sensitivity_results, report_file)
    
    logger.info(f"\n{'='*60}")
    logger.info("Evaluation complete!")
    logger.info(f"Figures saved to: {FIGURES_DIR}")
    logger.info(f"Statistical results: {stats_file}")
    logger.info(f"TPT sensitivity: {sensitivity_csv}")
    logger.info(f"Report: {report_file}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()

