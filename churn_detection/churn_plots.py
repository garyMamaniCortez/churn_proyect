"""EDA figures for the churn TRAINING dataset (churn_ciclos.csv).

Purpose
--------
This asks a specific question tied to the supervised problem: "what actually
separates a cycle that got renewed from one that churned". Distributions are
split by `estado_ciclo` wherever that adds insight, rather than shown in
aggregate -- an aggregate histogram can't tell you whether a feature is
predictive, a split one can.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

FEATURE_COLUMNS = [
    "tenure_dias",
    "porcentaje_uso_membresia",
    "recencia_dias",
    "frecuencia_visitas_semanal",
    "hora_promedio_checkin",
    "pct_visitas_fin_de_semana",
    "monto_total_gastado",
    "n_inscripciones_total",
    "ratio_actividad_reciente",
    "cv_gap_visitas",
    "monto_gastado_ultimos_90d",
]


class ChurnEDAFigureGenerator:
    """Renders a fixed set of EDA figures for the per-cycle churn dataset."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, df: pd.DataFrame) -> list[Path]:
        generators = [
            self._plot_estado_ciclo_counts,
            self._plot_missingness,
            self._plot_feature_distributions_by_label,
            self._plot_correlation_heatmap,
        ]
        return [gen(df) for gen in generators]

    # -- private helpers -------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        path = self._output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_estado_ciclo_counts(self, df: pd.DataFrame) -> Path:
        counts = df["estado_ciclo"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x=counts.index,
            y=counts.to_numpy(),
            hue=counts.index,
            palette={"renovado": "#55A868", "churned": "#C44E52", "censurado": "#8C8C8C"},
            legend=False,
            ax=ax,
        )
        for i, v in enumerate(counts.to_numpy()):
            ax.text(i, v, f"{v}\n({v / len(df):.1%})", ha="center", va="bottom")
        ax.set_ylabel("# ciclos de membresía")
        ax.set_ylim(0, counts.max() * 1.18)
        ax.set_title("Resultado por ciclo (churn_ciclos.csv)")
        return self._save(fig, "08_churn_estado_ciclo.png")

    def _plot_missingness(self, df: pd.DataFrame) -> Path:
        cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        null_pct = (df[cols].isna().mean() * 100).sort_values(ascending=False)
        null_pct = null_pct[null_pct > 0]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * max(len(null_pct), 1))))
        if len(null_pct) > 0:
            sns.barplot(x=null_pct.to_numpy(), y=null_pct.index, ax=ax, color="#4C72B0")
        ax.axvline(
            40, color="crimson", linestyle="--", linewidth=1, label="Umbral 40% (se elimina)"
        )
        ax.set_xlabel("% de valores nulos")
        ax.set_ylabel("")
        ax.set_title("Nulos por columna — churn_ciclos.csv")
        ax.legend()
        return self._save(fig, "09_churn_missingness.png")

    def _plot_feature_distributions_by_label(self, df: pd.DataFrame) -> Path:
        labeled = df[df["estado_ciclo"].isin(["renovado", "churned"])]
        cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        ncols = 3
        nrows = -(-len(cols) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        palette = {"renovado": "#55A868", "churned": "#C44E52"}
        for ax, col in zip(axes_flat, cols):
            sns.kdeplot(
                data=labeled,
                x=col,
                hue="estado_ciclo",
                palette=palette,
                common_norm=False,
                fill=True,
                alpha=0.3,
                ax=ax,
                warn_singular=False,
            )
            ax.set_title(col)
            ax.set_xlabel("")
        for ax in axes_flat[len(cols) :]:
            ax.axis("off")
        fig.suptitle("Distribución por feature: renovado vs. churned", y=1.02)
        return self._save(fig, "10_churn_distribuciones_por_label.png")

    def _plot_correlation_heatmap(self, df: pd.DataFrame) -> Path:
        cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        corr = df[cols].corr(numeric_only=True)
        size = max(0.6 * len(cols) + 2, 4)
        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": 7},
        )
        ax.set_title("Correlación entre features — churn_ciclos.csv")
        return self._save(fig, "11_churn_correlacion.png")


# --- CLI ---

if __name__ == "__main__":
    import typer

    from churn_detection.config import FIGURES_DIR, PROCESSED_DATA_DIR

    app = typer.Typer(help="Generate EDA figures for the churn training dataset.")

    @app.command()
    def generate_churn_eda_figures(
        input_path: Path = PROCESSED_DATA_DIR / "churn_ciclos.csv",
        output_dir: Path = FIGURES_DIR,
    ) -> None:
        from loguru import logger

        df = pd.read_csv(input_path)
        generator = ChurnEDAFigureGenerator(output_dir)
        paths = generator.generate_all(df)
        for path in paths:
            logger.success(f"Wrote {path}")

    app()
