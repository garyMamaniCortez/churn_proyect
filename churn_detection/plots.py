"""EDA figures for the client segmentation dataset.

Design notes (SOLID)
---------------------
`EDAFigureGenerator` takes an already-loaded DataFrame and an output directory
-- it doesn't read CSVs or know about DVC/paths beyond where to write PNGs
(SRP / Dependency Inversion), so it's testable with a small synthetic
DataFrame and a `tmp_path` fixture, no real dataset required.

Each `_plot_*` method renders exactly one figure, saves it, closes it (so
matplotlib doesn't accumulate open figures across calls), and returns the path
it wrote to. `generate_all()` calls each and collects the paths. Missing
columns are skipped gracefully rather than raising, so this keeps working as
the segmentation table's column set evolves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display available, and we only ever save to disk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

NUMERIC_FEATURES_FOR_DISTRIBUTION = [
    "tenure_dias",
    "porcentaje_uso_membresia",
    "recencia_dias",
    "frecuencia_visitas_semanal",
    "hora_promedio_checkin",
    "monto_total_gastado",
]

CORRELATION_FEATURES = [
    "tenure_dias",
    "n_inscripciones_total",
    "porcentaje_uso_membresia",
    "n_checkins_total",
    "n_checkins_ultimos_30d",
    "recencia_dias",
    "frecuencia_visitas_semanal",
    "hora_promedio_checkin",
    "hora_checkin_std",
    "pct_visitas_fin_de_semana",
    "monto_total_gastado",
    "monto_promedio_venta",
]

_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class EDAFigureGenerator:
    """Renders a fixed set of EDA figures for the client segmentation table."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, df: pd.DataFrame) -> list[Path]:
        generators = [
            self._plot_missingness,
            self._plot_numeric_distributions,
            self._plot_correlation_heatmap,
            self._plot_categorical_breakdowns,
            self._plot_usage_vs_recency_scatter,
        ]
        return [gen(df) for gen in generators]

    # -- private helpers -------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        path = self._output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_missingness(self, df: pd.DataFrame) -> Path:
        null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        null_pct = null_pct[null_pct > 0]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * max(len(null_pct), 1))))
        if len(null_pct) > 0:
            sns.barplot(x=null_pct.to_numpy(), y=null_pct.index, ax=ax, color="#4C72B0")
        ax.axvline(
            40, color="crimson", linestyle="--", linewidth=1, label="Umbral 40% (se elimina)"
        )
        ax.set_xlabel("% de valores nulos")
        ax.set_ylabel("")
        ax.set_title("Nulos por columna — clientes_segmentacion.csv")
        ax.legend()
        return self._save(fig, "01_missingness.png")

    def _plot_numeric_distributions(self, df: pd.DataFrame) -> Path:
        cols = [c for c in NUMERIC_FEATURES_FOR_DISTRIBUTION if c in df.columns]
        ncols = 3
        nrows = -(-len(cols) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        for ax, col in zip(axes_flat, cols):
            sns.histplot(df[col].dropna(), bins=30, ax=ax, color="#55A868")
            ax.set_title(col)
            ax.set_xlabel("")
        for ax in axes_flat[len(cols) :]:
            ax.axis("off")
        fig.suptitle("Distribuciones de features numéricas clave", y=1.02)
        return self._save(fig, "02_distribuciones_numericas.png")

    def _plot_correlation_heatmap(self, df: pd.DataFrame) -> Path:
        cols = [c for c in CORRELATION_FEATURES if c in df.columns]
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
        ax.set_title("Correlación entre features numéricas")
        return self._save(fig, "03_correlacion.png")

    def _plot_categorical_breakdowns(self, df: pd.DataFrame) -> Path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        if "dia_semana_mas_frecuente" in df.columns:
            counts = df["dia_semana_mas_frecuente"].value_counts().reindex(_DAY_ORDER).fillna(0)
            sns.barplot(x=counts.to_numpy(), y=counts.index, ax=axes[0], color="#C44E52")
            axes[0].set_title("Día de semana más frecuente")

        if "es_multisucursal" in df.columns:
            counts = df["es_multisucursal"].value_counts()
            sns.barplot(
                x=counts.index.astype(str), y=counts.to_numpy(), ax=axes[1], color="#8172B2"
            )
            axes[1].set_title("Usa más de una sucursal")

        if "tiene_pago_pendiente" in df.columns:
            counts = df["tiene_pago_pendiente"].value_counts()
            sns.barplot(
                x=counts.index.astype(str), y=counts.to_numpy(), ax=axes[2], color="#CCB974"
            )
            axes[2].set_title("Tiene pago pendiente")

        fig.suptitle("Variables categóricas / binarias")
        return self._save(fig, "04_categoricas.png")

    def _plot_usage_vs_recency_scatter(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(7, 6))
        required = ["porcentaje_uso_membresia", "recencia_dias"]
        if all(c in df.columns for c in required):
            plot_df = df.dropna(subset=required)
            extra_kwargs = {}
            if "monto_total_gastado" in plot_df.columns:
                extra_kwargs["size"] = "monto_total_gastado"
            if "frecuencia_visitas_semanal" in plot_df.columns:
                extra_kwargs["hue"] = "frecuencia_visitas_semanal"
            sns.scatterplot(
                data=plot_df,
                x="recencia_dias",
                y="porcentaje_uso_membresia",
                palette="viridis",
                alpha=0.6,
                ax=ax,
                **extra_kwargs,
            )
            if extra_kwargs:
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.set_title(
            "Uso de membresía vs. recencia\n(tamaño=gasto total, color=frecuencia semanal)"
        )
        ax.set_xlabel("Días desde el último check-in")
        ax.set_ylabel("% de membresía usado")
        return self._save(fig, "05_uso_vs_recencia.png")


# --- CLI: orchestrates reading data/processed/clientes_segmentacion.csv and writing reports/figures/ ---

if __name__ == "__main__":
    import typer

    from churn_detection.config import FIGURES_DIR, PROCESSED_DATA_DIR

    app = typer.Typer(help="Generate EDA figures for the client segmentation dataset.")

    @app.command()
    def generate_eda_figures(
        input_path: Path = PROCESSED_DATA_DIR / "clientes_segmentacion.csv",
        output_dir: Path = FIGURES_DIR,
    ) -> None:
        from loguru import logger

        df = pd.read_csv(input_path)
        generator = EDAFigureGenerator(output_dir)
        paths = generator.generate_all(df)
        for path in paths:
            logger.success(f"Wrote {path}")

    app()
