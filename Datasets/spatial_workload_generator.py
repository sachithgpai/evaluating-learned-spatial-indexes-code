from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except Exception:
    plt = None
    Rectangle = None


SELECTIVITY_SCALE = 1_000_000
DEFAULT_TARGET_FRACTIONS = [
    64 / SELECTIVITY_SCALE,
    256 / SELECTIVITY_SCALE,
    1024 / SELECTIVITY_SCALE,
    4096 / SELECTIVITY_SCALE,
    16384 / SELECTIVITY_SCALE,
]
PROJECT_CONFIG_FILENAME = "experiment_config.json"


def default_project_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / PROJECT_CONFIG_FILENAME


def load_project_config(config_path: Optional[str]) -> dict[str, Any]:
    env_config_path = os.environ.get("EXPERIMENT_CONFIG")
    path = Path(config_path or env_config_path or default_project_config_path())
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a JSON object: {path}")
    return config


def get_experiment_config(
    project_config: dict[str, Any],
    experiment_name: str,
) -> dict[str, Any]:
    experiments = project_config.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("Experiment config field 'experiments' must be an object.")

    if experiment_name not in experiments:
        known = ", ".join(sorted(experiments))
        raise ValueError(
            f"Unknown experiment '{experiment_name}' in experiment config. "
            f"Known experiments: {known}"
        )

    experiment_config = experiments[experiment_name]
    if not isinstance(experiment_config, dict):
        raise ValueError(
            f"Experiment config entry '{experiment_name}' must be an object."
        )
    return experiment_config


def config_value(config: dict[str, Any], config_name: str, fallback: Any) -> Any:
    return config.get(config_name, fallback)


@dataclass
class GeneratorConfig:
    # Common
    seed: Optional[int] = 42
    n_queries: int = 200
    num_query_scales: int = 5
    num_query_clusters: int = 5
    target_fractions: list[float] = field(
        default_factory=lambda: list(DEFAULT_TARGET_FRACTIONS)
    )

    # Synthetic dataset
    synthetic_n_points: int = 100_000
    synthetic_num_datasets: int = 5
    synthetic_num_clusters: int = 10
    a: float = 0.0001
    b: float = 0.003

    # Count-based query generation
    density_grid_size: int = 512
    center_grid_size: int = 256
    max_box_search_steps: int = 30
    min_halfwidth: float = 1e-4
    max_halfwidth: float = 0.5
    exact_refine_steps: int = 0

    # Real-world bbox sampling from parquet
    world_lat_step: float = 0.05
    world_lon_step: float = 0.1
    world_lat_min: float = -90.0
    world_lat_max: float = 90.0
    world_lon_min: float = -180.0
    world_lon_max: float = 180.0
    parquet_batch_rows: int = 2_000_000
    real_target_points: int = 8_000_000
    approx_count_low: int = 12_000_000
    approx_count_high: int = 16_000_000
    bbox_min_width_deg: float = 5.0
    bbox_max_width_deg: float = 30.0
    bbox_min_height_deg: float = 5.0
    bbox_max_height_deg: float = 30.0
    real_num_samples: int = 5
    real_max_bbox_tries: int = 100

    # Real-world KNN centroid generation
    real_knn_k: int = 512
    real_center_candidates: int = 2000

    # Output / plotting
    plot_sample_limit: int = 100_000
    plot_alpha: float = 0.1
    plot_size: float = 0.1
    query_plot_max_rectangles: int = 200
    query_rect_linewidth: float = 0.8
    query_rect_alpha: float = 0.35


class SpatialWorkloadGenerator:
    """Generate synthetic or real spatial workloads for the experiment runner."""

    def __init__(self, config: GeneratorConfig, output_root: Union[str, Path]):
        self.cfg = config
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.scaler_matrix = (self.cfg.b - self.cfg.a) * np.ones((2, 2)) + self.cfg.a
        self.synthetic_scales = np.flip(
            np.logspace(0.1, 1.4, num=self.cfg.num_query_scales)
        )
        self.query_scales = np.flip(
            np.logspace(0.1, 1.4, num=self.cfg.num_query_scales)
        )

        self.n_lat = int(
            np.ceil(
                (self.cfg.world_lat_max - self.cfg.world_lat_min)
                / self.cfg.world_lat_step
            )
        )
        self.n_lon = int(
            np.ceil(
                (self.cfg.world_lon_max - self.cfg.world_lon_min)
                / self.cfg.world_lon_step
            )
        )

    def log(self, message: str) -> None:
        print(message, flush=True)

    # ------------------------------------------------------------------
    # Generic utilities
    # ------------------------------------------------------------------

    def entropy(self, points: np.ndarray, nbins: int = 512) -> float:
        hist = np.histogram2d(points[:, 0], points[:, 1], bins=nbins)[0]
        hist_prob = np.ravel(hist / max(points.shape[0], 1))
        hist_prob = np.where(hist_prob < 1e-9, 1e-9, hist_prob)
        return float(-np.sum(hist_prob * np.log2(hist_prob)))

    def ensure_unit_square(self, points: np.ndarray) -> np.ndarray:
        return np.clip(points, 0.0, 1.0)

    def save_points(
        self,
        path: Path,
        points: np.ndarray,
        fmt: str = "%.9f",
        delimiter: str = " ",
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, points, delimiter=delimiter, fmt=fmt)

    def save_queries(
        self,
        path: Path,
        queries: np.ndarray,
        fmt: str = "%.9f",
        delimiter: str = " ",
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, queries, delimiter=delimiter, fmt=fmt)

    def save_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def save_plot(
        self,
        path: Path,
        points: np.ndarray,
        title: Optional[str] = None,
    ) -> None:
        if plt is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        pts = points[: self.cfg.plot_sample_limit]
        plt.figure(figsize=(4, 4), dpi=250)
        plt.scatter(pts[:, 0], pts[:, 1], alpha=self.cfg.plot_alpha, s=self.cfg.plot_size)
        if title:
            plt.title(title)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(path, dpi=250)
        plt.close()

    def save_query_overlay_plot(
        self,
        path: Path,
        points: np.ndarray,
        queries: np.ndarray,
        title: Optional[str] = None,
    ) -> None:
        if plt is None or Rectangle is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        pts = points[: self.cfg.plot_sample_limit]
        fig, ax = plt.subplots(figsize=(4, 4), dpi=250)
        ax.scatter(pts[:, 0], pts[:, 1], alpha=self.cfg.plot_alpha, s=self.cfg.plot_size)

        for query in queries[: self.cfg.query_plot_max_rectangles]:
            xmin, ymin, xmax, ymax = query
            rect = Rectangle(
                (xmin, ymin),
                max(0.0, xmax - xmin),
                max(0.0, ymax - ymin),
                fill=False,
                linewidth=self.cfg.query_rect_linewidth,
                alpha=self.cfg.query_rect_alpha,
            )
            ax.add_patch(rect)

        if title:
            ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        fig.savefig(path, dpi=250)
        plt.close(fig)

    def frac_to_tag(self, frac: float) -> str:
        scaled = int(round(frac * SELECTIVITY_SCALE))
        if math.isclose(
            frac,
            scaled / SELECTIVITY_SCALE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return f"{scaled:05d}"
        return f"{frac:.8f}".rstrip("0").rstrip(".").replace(".", "p")

    def area_halfwidth(self, frac: float) -> float:
        return math.sqrt(max(frac, 0.0)) / 2.0

    def build_area_queries(self, centers: np.ndarray, frac: float) -> np.ndarray:
        halfwidth = self.area_halfwidth(frac)
        x = centers[:, 0]
        y = centers[:, 1]
        return np.stack(
            [
                np.clip(x - halfwidth, 0.0, 1.0),
                np.clip(y - halfwidth, 0.0, 1.0),
                np.clip(x + halfwidth, 0.0, 1.0),
                np.clip(y + halfwidth, 0.0, 1.0),
            ],
            axis=-1,
        )

    def write_entropy_table(
        self,
        path: Path,
        rows: np.ndarray,
        fmt: Union[str, list[str]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, rows, fmt=fmt)

    # ------------------------------------------------------------------
    # Synthetic dataset generation
    # ------------------------------------------------------------------

    def random_covariance(self) -> np.ndarray:
        matrix = self.rng.random((2, 2)) * 2.0 - 1.0
        return np.multiply(matrix @ matrix.T, self.scaler_matrix)

    def sample_truncated_gaussian(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        count: int,
    ) -> np.ndarray:
        x = self.rng.random((max(1, int(count * 0.0001)), 2))
        while x.shape[0] < count:
            sample = self.rng.multivariate_normal(mean, cov, max(1000, count))
            mask = (
                (sample[:, 0] > 0.0)
                & (sample[:, 0] < 1.0)
                & (sample[:, 1] > 0.0)
                & (sample[:, 1] < 1.0)
            )
            x = np.vstack((x, sample[mask]))
        return x[:count]

    def generate_synthetic_gmm_parameters(
        self,
        num_clusters: Optional[int] = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        num_clusters = num_clusters or self.cfg.synthetic_num_clusters
        means = 0.9 * self.rng.random((num_clusters, 2)) + 0.05
        covariances = [self.random_covariance() for _ in range(num_clusters)]
        return means, covariances

    def generate_synthetic_dataset_from_gmm(
        self,
        means: np.ndarray,
        covariances: list[np.ndarray],
        n_points: Optional[int] = None,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, dict]:
        n_points = n_points or self.cfg.synthetic_n_points
        num_clusters = len(means)
        num_points_per_cluster = (n_points // num_clusters) + 1

        points = self.rng.random((10_000, 2))
        for cluster_id in range(num_clusters):
            cov = np.array(covariances[cluster_id], copy=True)
            cov[0, 0] *= scale
            cov[1, 1] *= scale
            points = np.vstack(
                (
                    points,
                    self.sample_truncated_gaussian(
                        means[cluster_id],
                        cov,
                        num_points_per_cluster,
                    ),
                )
            )

        self.rng.shuffle(points, axis=0)
        points = self.ensure_unit_square(points[:n_points])
        metadata = {
            "n_points": int(n_points),
            "num_clusters": int(num_clusters),
            "scale": float(scale),
            "entropy": self.entropy(points),
        }
        return points, metadata

    def generate_synthetic_query_centers(
        self,
        data_points: np.ndarray,
        query_scale: float,
        n_queries: Optional[int] = None,
        num_query_clusters: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        n_queries = n_queries or self.cfg.n_queries
        num_query_clusters = num_query_clusters or self.cfg.num_query_clusters
        num_query_clusters = max(1, min(num_query_clusters, len(data_points)))
        per_cluster = (n_queries // num_query_clusters) + 1

        mean_indices = self.rng.choice(
            len(data_points),
            num_query_clusters,
            replace=False,
        )
        means = data_points[mean_indices]
        covariances = [self.random_covariance() for _ in range(num_query_clusters)]

        queries = []
        for cluster_id in range(num_query_clusters):
            cov = covariances[cluster_id].copy()
            cov[0, 0] *= query_scale
            cov[1, 1] *= query_scale
            queries.append(
                self.sample_truncated_gaussian(means[cluster_id], cov, per_cluster)
            )

        queries = np.vstack(queries)
        self.rng.shuffle(queries, axis=0)
        queries = self.ensure_unit_square(queries[:n_queries])
        metadata = {
            "query_scale": float(query_scale),
            "num_query_clusters": int(num_query_clusters),
            "entropy": self.entropy(queries, nbins=32),
        }
        return queries, metadata

    # ------------------------------------------------------------------
    # Count-based query generation
    # ------------------------------------------------------------------

    def build_density_grid(
        self,
        points: np.ndarray,
        grid_size: Optional[int] = None,
    ) -> np.ndarray:
        grid_size = grid_size or self.cfg.density_grid_size
        ix = np.clip((points[:, 0] * grid_size).astype(np.int64), 0, grid_size - 1)
        iy = np.clip((points[:, 1] * grid_size).astype(np.int64), 0, grid_size - 1)
        flat = ix * grid_size + iy
        counts = np.bincount(flat, minlength=grid_size * grid_size)
        return counts.reshape(grid_size, grid_size)

    def build_prefix_sum(self, grid: np.ndarray) -> np.ndarray:
        return grid.cumsum(axis=0).cumsum(axis=1)

    def estimate_count_in_rect(
        self,
        prefix: np.ndarray,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> int:
        grid_size = prefix.shape[0]
        x0 = max(0, min(grid_size - 1, int(np.floor(xmin * grid_size))))
        y0 = max(0, min(grid_size - 1, int(np.floor(ymin * grid_size))))
        x1 = max(0, min(grid_size - 1, int(np.floor(xmax * grid_size))))
        y1 = max(0, min(grid_size - 1, int(np.floor(ymax * grid_size))))
        if x1 < x0 or y1 < y0:
            return 0

        total = prefix[x1, y1]
        if x0 > 0:
            total -= prefix[x0 - 1, y1]
        if y0 > 0:
            total -= prefix[x1, y0 - 1]
        if x0 > 0 and y0 > 0:
            total += prefix[x0 - 1, y0 - 1]
        return int(total)

    def exact_count_in_rect(self, points: np.ndarray, rect: np.ndarray) -> int:
        xmin, ymin, xmax, ymax = rect
        mask = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
        )
        return int(mask.sum())

    def estimate_local_aspect_ratio(
        self,
        points: np.ndarray,
        center: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[float, float]:
        local_points = points[indices]
        if len(local_points) == 0:
            return 1e-3, 1e-3
        varx = float(np.var(local_points[:, 0])) + 1e-6
        vary = float(np.var(local_points[:, 1])) + 1e-6
        return math.sqrt(varx), math.sqrt(vary)

    def find_anisotropic_box_for_target_count(
        self,
        prefix: np.ndarray,
        center: np.ndarray,
        target_count: int,
        sx: float,
        sy: float,
    ) -> np.ndarray:
        cx = float(center[0])
        cy = float(center[1])
        low = 1e-3
        high = 50.0
        best_box = None
        best_error = float("inf")

        for _ in range(self.cfg.max_box_search_steps):
            alpha = 0.5 * (low + high)
            halfwidth_x = np.clip(
                alpha * sx,
                self.cfg.min_halfwidth,
                self.cfg.max_halfwidth,
            )
            halfwidth_y = np.clip(
                alpha * sy,
                self.cfg.min_halfwidth,
                self.cfg.max_halfwidth,
            )
            xmin = max(0.0, cx - halfwidth_x)
            ymin = max(0.0, cy - halfwidth_y)
            xmax = min(1.0, cx + halfwidth_x)
            ymax = min(1.0, cy + halfwidth_y)

            estimate = self.estimate_count_in_rect(prefix, xmin, ymin, xmax, ymax)
            error = abs(estimate - target_count)
            if error < best_error:
                best_error = error
                best_box = np.array([xmin, ymin, xmax, ymax], dtype=np.float64)

            if estimate < target_count:
                low = alpha
            else:
                high = alpha

        return best_box

    def refine_box_exact(
        self,
        points: np.ndarray,
        center: np.ndarray,
        initial_box: np.ndarray,
        target_count: int,
        steps: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        steps = self.cfg.exact_refine_steps if steps is None else steps
        if steps <= 0:
            return initial_box, self.exact_count_in_rect(points, initial_box)

        cx = float(center[0])
        cy = float(center[1])
        xmin, ymin, xmax, ymax = initial_box
        halfwidth_x = max(cx - xmin, xmax - cx)
        halfwidth_y = max(cy - ymin, ymax - cy)

        best_box = initial_box.copy()
        best_count = self.exact_count_in_rect(points, best_box)
        best_error = abs(best_count - target_count)
        low = 0.5
        high = 1.5

        for _ in range(steps):
            alpha = 0.5 * (low + high)
            candidate = np.array(
                [
                    max(0.0, cx - halfwidth_x * alpha),
                    max(0.0, cy - halfwidth_y * alpha),
                    min(1.0, cx + halfwidth_x * alpha),
                    min(1.0, cy + halfwidth_y * alpha),
                ],
                dtype=np.float64,
            )
            count = self.exact_count_in_rect(points, candidate)
            error = abs(count - target_count)
            if error < best_error:
                best_error = error
                best_box = candidate
                best_count = count
            if count < target_count:
                low = alpha
            else:
                high = alpha

        return best_box, int(best_count)

    # ------------------------------------------------------------------
    # Real-world helpers
    # ------------------------------------------------------------------

    def build_world_count_grid(
        self,
        parquet_path: str,
        batch_rows: Optional[int] = None,
        dtype=np.int64,
    ) -> np.ndarray:
        if pq is None:
            raise ImportError("pyarrow is required for parquet support.")
        batch_rows = batch_rows or self.cfg.parquet_batch_rows
        parquet_file = pq.ParquetFile(parquet_path)
        counts_flat = np.zeros(self.n_lat * self.n_lon, dtype=dtype)

        for batch in parquet_file.iter_batches(
            batch_size=batch_rows,
            columns=["lat", "lon"],
        ):
            lat = batch.column(0).to_numpy(zero_copy_only=False)
            lon = batch.column(1).to_numpy(zero_copy_only=False)
            i = np.floor((lat - self.cfg.world_lat_min) / self.cfg.world_lat_step).astype(
                np.int64
            )
            j = np.floor((lon - self.cfg.world_lon_min) / self.cfg.world_lon_step).astype(
                np.int64
            )
            mask = (i >= 0) & (i < self.n_lat) & (j >= 0) & (j < self.n_lon)
            if not np.any(mask):
                continue
            indices = i[mask] * self.n_lon + j[mask]
            counts_flat += np.bincount(indices, minlength=self.n_lat * self.n_lon).astype(
                dtype
            )

        return counts_flat.reshape(self.n_lat, self.n_lon)

    def save_world_grid(self, path_npz: Union[str, Path], grid: np.ndarray) -> None:
        np.savez_compressed(
            path_npz,
            grid=grid,
            lat_min=self.cfg.world_lat_min,
            lon_min=self.cfg.world_lon_min,
            lat_step=self.cfg.world_lat_step,
            lon_step=self.cfg.world_lon_step,
            n_lat=self.n_lat,
            n_lon=self.n_lon,
        )

    def load_world_grid(self, path_npz: Union[str, Path]) -> np.ndarray:
        data = np.load(path_npz)
        return data["grid"]

    def approx_count_bbox(
        self,
        grid: np.ndarray,
        lat0: float,
        lon0: float,
        lat1: float,
        lon1: float,
    ) -> int:
        if lat0 > lat1:
            lat0, lat1 = lat1, lat0
        if lon0 > lon1:
            lon0, lon1 = lon1, lon0

        i0 = int(np.floor((lat0 - self.cfg.world_lat_min) / self.cfg.world_lat_step))
        i1 = int(np.floor((lat1 - self.cfg.world_lat_min) / self.cfg.world_lat_step))
        j0 = int(np.floor((lon0 - self.cfg.world_lon_min) / self.cfg.world_lon_step))
        j1 = int(np.floor((lon1 - self.cfg.world_lon_min) / self.cfg.world_lon_step))

        i0 = max(0, min(self.n_lat - 1, i0))
        i1 = max(0, min(self.n_lat - 1, i1))
        j0 = max(0, min(self.n_lon - 1, j0))
        j1 = max(0, min(self.n_lon - 1, j1))
        return int(grid[i0 : i1 + 1, j0 : j1 + 1].sum())

    def make_random_world_bbox(self) -> tuple[float, float, float, float]:
        u = self.rng.uniform(-1.0, 1.0)
        center_lat = np.degrees(np.arcsin(u))
        center_lon = self.rng.uniform(-180.0, 180.0)

        width = self.rng.uniform(self.cfg.bbox_min_width_deg, self.cfg.bbox_max_width_deg)
        height = self.rng.uniform(
            self.cfg.bbox_min_height_deg,
            self.cfg.bbox_max_height_deg,
        )
        half_width = width / 2.0
        half_height = height / 2.0

        min_lat = max(-90.0, center_lat - half_height)
        max_lat = min(90.0, center_lat + half_height)
        min_lon = center_lon - half_width
        max_lon = center_lon + half_width

        if min_lon < -180.0:
            shift = -180.0 - min_lon
            min_lon += shift
            max_lon += shift
        if max_lon > 180.0:
            shift = max_lon - 180.0
            min_lon -= shift
            max_lon -= shift

        min_lon = max(-180.0, min_lon)
        max_lon = min(180.0, max_lon)
        return min_lat, min_lon, max_lat, max_lon

    def filter_points_bbox(
        self,
        parquet_path: str,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        batch_rows: Optional[int] = None,
    ) -> np.ndarray:
        if pq is None:
            raise ImportError("pyarrow is required for parquet support.")
        batch_rows = batch_rows or self.cfg.parquet_batch_rows
        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon

        parquet_file = pq.ParquetFile(parquet_path)
        lat_chunks: list[np.ndarray] = []
        lon_chunks: list[np.ndarray] = []

        for batch in parquet_file.iter_batches(
            batch_size=batch_rows,
            columns=["lat", "lon"],
        ):
            lat = batch.column(0).to_numpy(zero_copy_only=False)
            lon = batch.column(1).to_numpy(zero_copy_only=False)
            mask = (
                (lat >= min_lat)
                & (lat <= max_lat)
                & (lon >= min_lon)
                & (lon <= max_lon)
            )
            if np.any(mask):
                lat_chunks.append(lat[mask])
                lon_chunks.append(lon[mask])

        if not lat_chunks:
            return np.empty((0, 2), dtype=np.float32)

        lat_all = np.concatenate(lat_chunks)
        lon_all = np.concatenate(lon_chunks)
        return np.column_stack((lat_all, lon_all)).astype(np.float32, copy=False)

    def normalize_points_to_unit_square(
        self,
        points: np.ndarray,
        eps: float = 1e-9,
    ) -> tuple[np.ndarray, dict]:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        denom = np.maximum(maxs - mins, eps)
        normalized = (points - mins) / denom
        return self.ensure_unit_square(normalized), {
            "mins": mins.tolist(),
            "maxs": maxs.tolist(),
        }

    def sample_real_dataset_from_bbox(
        self,
        parquet_path: str,
        world_grid: np.ndarray,
        target_count: Optional[int] = None,
        approx_low: Optional[int] = None,
        approx_high: Optional[int] = None,
        max_tries: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        target_count = target_count or self.cfg.real_target_points
        approx_low = approx_low or self.cfg.approx_count_low
        approx_high = approx_high or self.cfg.approx_count_high
        max_tries = max_tries or self.cfg.real_max_bbox_tries

        for attempt in range(1, max_tries + 1):
            bbox = self.make_random_world_bbox()
            approx = self.approx_count_bbox(world_grid, *bbox)
            if not (approx_low <= approx <= approx_high):
                continue

            raw_points = self.filter_points_bbox(parquet_path, *bbox)
            actual_count = len(raw_points)
            if actual_count < target_count:
                continue

            self.rng.shuffle(raw_points, axis=0)
            raw_points = raw_points[:target_count]
            normalized, norm_meta = self.normalize_points_to_unit_square(raw_points)
            return normalized, {
                "bbox": list(map(float, bbox)),
                "approx_count": int(approx),
                "actual_count_before_trim": int(actual_count),
                "target_count": int(target_count),
                "attempt": int(attempt),
                **norm_meta,
            }

        raise RuntimeError("Could not find a suitable real-world bbox sample.")

    # ------------------------------------------------------------------
    # Real-world KNN centroid generation
    # ------------------------------------------------------------------

    def build_center_cell_index(self, points: np.ndarray) -> dict:
        grid_size = self.cfg.center_grid_size
        ix = np.clip((points[:, 0] * grid_size).astype(np.int32), 0, grid_size - 1)
        iy = np.clip((points[:, 1] * grid_size).astype(np.int32), 0, grid_size - 1)
        cell_ids = ix.astype(np.int64) * grid_size + iy.astype(np.int64)
        order = np.argsort(cell_ids, kind="mergesort")
        sorted_ids = cell_ids[order]
        unique_ids, starts, counts = np.unique(
            sorted_ids,
            return_index=True,
            return_counts=True,
        )
        return {
            "grid_size": grid_size,
            "order": order,
            "unique_ids": unique_ids,
            "starts": starts,
            "counts": counts,
        }

    def _cell_points_indices(self, index: dict, cell_id: int) -> np.ndarray:
        position = np.searchsorted(index["unique_ids"], cell_id)
        if (
            position >= len(index["unique_ids"])
            or index["unique_ids"][position] != cell_id
        ):
            return np.empty(0, dtype=np.int64)
        start = index["starts"][position]
        count = index["counts"][position]
        return index["order"][start : start + count]

    def local_knn_indices(
        self,
        points: np.ndarray,
        center: np.ndarray,
        k: int,
        index: dict,
    ) -> np.ndarray:
        grid_size = index["grid_size"]
        cx = max(0, min(grid_size - 1, int(center[0] * grid_size)))
        cy = max(0, min(grid_size - 1, int(center[1] * grid_size)))

        gathered = []
        gathered_count = 0
        for radius in range(grid_size):
            x0 = max(0, cx - radius)
            x1 = min(grid_size - 1, cx + radius)
            y0 = max(0, cy - radius)
            y1 = min(grid_size - 1, cy + radius)

            for xi in range(x0, x1 + 1):
                for yi in range(y0, y1 + 1):
                    if radius > 0 and (x0 < xi < x1) and (y0 < yi < y1):
                        continue
                    cell_id = xi * grid_size + yi
                    indices = self._cell_points_indices(index, cell_id)
                    if len(indices) > 0:
                        gathered.append(indices)
                        gathered_count += len(indices)

            if gathered_count >= k or (
                x0 == 0
                and x1 == grid_size - 1
                and y0 == 0
                and y1 == grid_size - 1
            ):
                break

        if not gathered:
            return np.empty(0, dtype=np.int64)

        candidates = np.concatenate(gathered)
        if len(candidates) <= k:
            return candidates

        candidate_points = points[candidates]
        distances = np.sum((candidate_points - center) ** 2, axis=1)
        local = np.argpartition(distances, kth=k - 1)[:k]
        return candidates[local]

    def random_knn_centroids(
        self,
        points: np.ndarray,
        n_centers: int,
        k: Optional[int] = None,
        candidate_pool: Optional[int] = None,
        cell_index: Optional[dict] = None,
    ) -> tuple[np.ndarray, list[np.ndarray], dict]:
        k = k or self.cfg.real_knn_k
        candidate_pool = candidate_pool or self.cfg.real_center_candidates
        cell_index = cell_index or self.build_center_cell_index(points)

        n_points = len(points)
        candidate_pool = min(candidate_pool, n_points)
        n_centers = min(n_centers, candidate_pool)
        candidate_indices = self.rng.choice(n_points, size=candidate_pool, replace=False)
        seed_indices = self.rng.choice(candidate_pool, size=n_centers, replace=False)
        seeds = points[candidate_indices[seed_indices]]

        centroids = np.empty((n_centers, 2), dtype=np.float64)
        knn_indices_list: list[np.ndarray] = []
        for idx, seed in enumerate(seeds):
            knn_indices = self.local_knn_indices(points, seed, k=k, index=cell_index)
            if len(knn_indices) == 0:
                knn_indices = np.array([candidate_indices[seed_indices[idx]]], dtype=np.int64)
            knn_indices_list.append(knn_indices)
            centroids[idx] = points[knn_indices].mean(axis=0)

        metadata = {
            "k": int(k),
            "candidate_pool": int(candidate_pool),
            "n_centers": int(n_centers),
        }
        return self.ensure_unit_square(centroids), knn_indices_list, metadata

    def generate_real_query_centers(
        self,
        points: np.ndarray,
        query_scale: float,
        n_queries: Optional[int] = None,
        num_query_clusters: Optional[int] = None,
        cell_index: Optional[dict] = None,
    ) -> tuple[np.ndarray, list[np.ndarray], dict]:
        n_queries = n_queries or self.cfg.n_queries
        num_query_clusters = num_query_clusters or self.cfg.num_query_clusters
        num_query_clusters = max(1, min(num_query_clusters, len(points)))
        per_cluster = (n_queries // num_query_clusters) + 1
        cell_index = cell_index or self.build_center_cell_index(points)

        base_centers, knn_lists, metadata = self.random_knn_centroids(
            points=points,
            n_centers=num_query_clusters,
            k=self.cfg.real_knn_k,
            candidate_pool=self.cfg.real_center_candidates,
            cell_index=cell_index,
        )

        queries = []
        for cluster_id, center in enumerate(base_centers):
            sx, sy = self.estimate_local_aspect_ratio(points, center, knn_lists[cluster_id])
            cov = np.array(
                [
                    [max((sx * sx) * query_scale, 1e-6), 0.0],
                    [0.0, max((sy * sy) * query_scale, 1e-6)],
                ]
            )
            queries.append(self.sample_truncated_gaussian(center, cov, per_cluster))

        queries = np.vstack(queries)
        self.rng.shuffle(queries, axis=0)
        queries = self.ensure_unit_square(queries[:n_queries])
        metadata.update(
            {
                "query_scale": float(query_scale),
                "entropy": self.entropy(queries, nbins=32),
            }
        )
        return queries, knn_lists, metadata

    # ------------------------------------------------------------------
    # Unified query generation
    # ------------------------------------------------------------------

    def generate_other_queries(self, points: np.ndarray, mode: str) -> dict:
        density_grid = self.build_density_grid(
            points,
            grid_size=self.cfg.density_grid_size,
        )
        prefix = self.build_prefix_sum(density_grid)
        cell_index = self.build_center_cell_index(points)
        uniform_entropy = self.entropy(self.rng.random((self.cfg.n_queries, 2)), nbins=32)

        payload: dict = {
            "mode": mode,
            "n_points": int(len(points)),
            "density_grid_size": int(self.cfg.density_grid_size),
            "target_fractions": list(self.cfg.target_fractions),
            "query_groups": [],
        }

        for group_num, query_scale in enumerate(self.query_scales, start=1):
            if mode == "synthetic":
                centers, center_meta = self.generate_synthetic_query_centers(
                    data_points=points,
                    query_scale=float(query_scale),
                    n_queries=self.cfg.n_queries,
                    num_query_clusters=self.cfg.num_query_clusters,
                )
                knn_lists = None
            elif mode == "real":
                centers, knn_lists, center_meta = self.generate_real_query_centers(
                    points=points,
                    query_scale=float(query_scale),
                    n_queries=self.cfg.n_queries,
                    num_query_clusters=self.cfg.num_query_clusters,
                    cell_index=cell_index,
                )
            else:
                raise ValueError("mode must be 'synthetic' or 'real'")

            normalized_entropy = 0.0
            if uniform_entropy > 0.0:
                normalized_entropy = float(center_meta["entropy"] / uniform_entropy)

            group = {
                "group_num": int(group_num),
                "query_scale": float(query_scale),
                "normalized_entropy": normalized_entropy,
                "centers": centers,
                "center_meta": center_meta,
                "area_queries_by_fraction": {},
                "count_queries_by_fraction": {},
                "query_meta_by_fraction": {},
            }

            for frac in self.cfg.target_fractions:
                target_count = max(1, int(round(len(points) * frac)))
                area_queries = self.build_area_queries(centers, frac)
                count_queries = np.empty((len(centers), 4), dtype=np.float64)
                meta_rows = []

                for query_id, center in enumerate(centers):
                    if mode == "real" and knn_lists is not None and query_id < len(knn_lists):
                        indices = knn_lists[query_id]
                    else:
                        indices = self.local_knn_indices(
                            points,
                            center,
                            k=min(256, len(points)),
                            index=cell_index,
                        )

                    sx, sy = self.estimate_local_aspect_ratio(points, center, indices)
                    initial_box = self.find_anisotropic_box_for_target_count(
                        prefix=prefix,
                        center=center,
                        target_count=target_count,
                        sx=sx,
                        sy=sy,
                    )
                    estimated_count = self.estimate_count_in_rect(prefix, *initial_box)
                    final_box, exact_count = self.refine_box_exact(
                        points,
                        center,
                        initial_box,
                        target_count,
                    )
                    count_queries[query_id] = final_box
                    meta_rows.append(
                        [
                            query_id,
                            float(frac),
                            int(target_count),
                            int(estimated_count),
                            int(exact_count),
                            float(center[0]),
                            float(center[1]),
                        ]
                    )

                group["area_queries_by_fraction"][frac] = area_queries
                group["count_queries_by_fraction"][frac] = count_queries
                group["query_meta_by_fraction"][frac] = np.asarray(
                    meta_rows,
                    dtype=np.float64,
                )

            payload["query_groups"].append(group)

        return payload

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def write_other_queries(
        self,
        dataset_dir: Path,
        data_entropy_id: int,
        payload: dict,
        points: np.ndarray,
    ) -> list[list[float]]:
        query_root = dataset_dir / "queries" / "otherDist"
        plot_root = query_root / "plots"
        query_root.mkdir(parents=True, exist_ok=True)
        plot_root.mkdir(parents=True, exist_ok=True)

        entropy_rows: list[list[float]] = []
        for group in payload["query_groups"]:
            query_entropy_id = group["group_num"]
            centers = group["centers"]
            entropy_rows.append(
                [
                    float(data_entropy_id),
                    float(query_entropy_id),
                    float(group["normalized_entropy"]),
                ]
            )

            self.save_points(
                query_root / f"{data_entropy_id}_querycenters_{query_entropy_id}.csv",
                centers,
                delimiter=",",
            )

            for frac in payload["target_fractions"]:
                selectivity = self.frac_to_tag(frac)
                self.save_points(
                    query_root
                    / f"{data_entropy_id}_{selectivity}_querycenteres_{query_entropy_id}",
                    centers,
                )
                self.save_queries(
                    query_root
                    / f"{data_entropy_id}_{selectivity}_areabased_{query_entropy_id}",
                    group["area_queries_by_fraction"][frac],
                )
                self.save_queries(
                    query_root
                    / f"{data_entropy_id}_{selectivity}_countbased_{query_entropy_id}",
                    group["count_queries_by_fraction"][frac],
                )

                count_meta = group["query_meta_by_fraction"][frac]
                np.savetxt(
                    query_root
                    / f"{data_entropy_id}_{selectivity}_countbased_meta_{query_entropy_id}.txt",
                    count_meta,
                    fmt=["%d", "%.8f", "%d", "%d", "%d", "%.8f", "%.8f"],
                    header=(
                        "query_id target_fraction target_count estimated_count "
                        "exact_count center_x center_y"
                    ),
                    comments="",
                )
                self.save_query_overlay_plot(
                    plot_root
                    / f"{data_entropy_id}_{selectivity}_countbased_{query_entropy_id}.png",
                    points=points,
                    queries=group["count_queries_by_fraction"][frac],
                    title=(
                        f"data_entropy={data_entropy_id} "
                        f"query_entropy={query_entropy_id} "
                        f"selectivity={selectivity}"
                    ),
                )

            serializable = {
                "mode": payload["mode"],
                "n_points": payload["n_points"],
                "density_grid_size": payload["density_grid_size"],
                "target_fractions": payload["target_fractions"],
                "data_entropy_id": int(data_entropy_id),
                "query_entropy_id": int(query_entropy_id),
                "query_scale": group["query_scale"],
                "normalized_entropy": group["normalized_entropy"],
                "center_meta": group["center_meta"],
            }
            self.save_json(
                query_root / f"{data_entropy_id}_query_summary_{query_entropy_id}.json",
                serializable,
            )

        return entropy_rows

    # ------------------------------------------------------------------
    # High-level workflows
    # ------------------------------------------------------------------

    def run_synthetic(self) -> None:
        uniform_entropy = self.entropy(
            self.rng.random((self.cfg.synthetic_n_points, 2))
        )

        for data_sample_num in range(1, self.cfg.synthetic_num_datasets + 1):
            dataset_dir = self.output_root / str(data_sample_num)
            datapoint_root = dataset_dir / "datapoints"
            query_root = dataset_dir / "queries"
            (datapoint_root / "plots").mkdir(parents=True, exist_ok=True)
            query_root.mkdir(parents=True, exist_ok=True)

            shared_means, shared_covariances = self.generate_synthetic_gmm_parameters(
                num_clusters=self.cfg.synthetic_num_clusters
            )
            self.save_points(
                datapoint_root / "shared_cluster_centers.csv",
                shared_means,
                delimiter=",",
            )
            self.save_json(
                datapoint_root / "shared_cluster_covariances.json",
                {
                    "covariances": [
                        np.asarray(covariance).tolist()
                        for covariance in shared_covariances
                    ]
                },
            )

            data_entropy_rows: list[list[float]] = []
            query_entropy_rows: list[list[float]] = []
            for data_entropy_id, scale in enumerate(self.synthetic_scales, start=1):
                points, metadata = self.generate_synthetic_dataset_from_gmm(
                    means=shared_means,
                    covariances=shared_covariances,
                    n_points=self.cfg.synthetic_n_points,
                    scale=float(scale),
                )
                normalized_entropy = 0.0
                if uniform_entropy > 0.0:
                    normalized_entropy = float(metadata["entropy"] / uniform_entropy)
                data_entropy_rows.append([float(data_entropy_id), normalized_entropy])

                self.save_points(datapoint_root / f"{data_entropy_id}", points)
                self.save_plot(
                    datapoint_root / "plots" / f"{data_entropy_id}.png",
                    points,
                    title=(
                        f"sample={data_sample_num} "
                        f"data_entropy={data_entropy_id}"
                    ),
                )
                metadata["shared_gmm_dataset_id"] = int(data_sample_num)
                self.save_json(
                    datapoint_root / f"{data_entropy_id}_meta.json",
                    metadata,
                )

                query_payload = self.generate_other_queries(points, mode="synthetic")
                query_entropy_rows.extend(
                    self.write_other_queries(
                        dataset_dir,
                        data_entropy_id,
                        query_payload,
                        points,
                    )
                )

            self.write_entropy_table(
                datapoint_root / "entropy_values",
                np.asarray(data_entropy_rows, dtype=np.float64),
                fmt=["%d", "%.9f"],
            )
            self.write_entropy_table(
                query_root / "entropy_values",
                np.asarray(query_entropy_rows, dtype=np.float64),
                fmt=["%d", "%d", "%.9f"],
            )
            self.save_json(dataset_dir / "config.json", asdict(self.cfg))

    def run_real(
        self,
        parquet_path: str,
        world_grid_path: Optional[str] = None,
        build_world_grid: bool = False,
    ) -> None:
        if pq is None:
            raise ImportError("pyarrow is required for real/parquet mode.")

        if world_grid_path and Path(world_grid_path).exists() and not build_world_grid:
            world_grid = self.load_world_grid(world_grid_path)
        else:
            world_grid = self.build_world_count_grid(parquet_path)
            if world_grid_path:
                self.save_world_grid(world_grid_path, world_grid)

        uniform_entropy = self.entropy(
            self.rng.random((self.cfg.real_target_points, 2))
        )

        for data_sample_num in range(1, self.cfg.real_num_samples + 1):
            dataset_dir = self.output_root / str(data_sample_num)
            datapoint_root = dataset_dir / "datapoints"
            query_root = dataset_dir / "queries"
            (datapoint_root / "plots").mkdir(parents=True, exist_ok=True)
            query_root.mkdir(parents=True, exist_ok=True)

            points, metadata = self.sample_real_dataset_from_bbox(
                parquet_path=parquet_path,
                world_grid=world_grid,
                target_count=self.cfg.real_target_points,
                approx_low=self.cfg.approx_count_low,
                approx_high=self.cfg.approx_count_high,
                max_tries=self.cfg.real_max_bbox_tries,
            )

            data_entropy = self.entropy(points)
            normalized_entropy = 0.0
            if uniform_entropy > 0.0:
                normalized_entropy = float(data_entropy / uniform_entropy)

            self.save_points(datapoint_root / "1", points, fmt="%.8f")
            self.save_plot(
                datapoint_root / "plots" / "1.png",
                points,
                title=f"real sample={data_sample_num}",
            )
            metadata["entropy"] = data_entropy
            self.save_json(datapoint_root / "meta.json", metadata)

            query_payload = self.generate_other_queries(points, mode="real")
            query_entropy_rows = self.write_other_queries(
                dataset_dir,
                1,
                query_payload,
                points,
            )

            self.write_entropy_table(
                datapoint_root / "entropy_values",
                np.asarray([[1.0, normalized_entropy]], dtype=np.float64),
                fmt=["%d", "%.9f"],
            )
            self.write_entropy_table(
                query_root / "entropy_values",
                np.asarray(query_entropy_rows, dtype=np.float64),
                fmt=["%d", "%d", "%.9f"],
            )
            self.save_json(dataset_dir / "config.json", asdict(self.cfg))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic or real spatial workloads from the shared "
            "experiment config."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    synthetic = subparsers.add_parser(
        "synthetic",
        help="Generate synthetic datasets plus area-based and count-based queries.",
    )
    synthetic.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the project experiment config JSON. Defaults to "
            "../experiment_config.json or EXPERIMENT_CONFIG."
        ),
    )
    synthetic.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment key inside the config. Defaults to 'synthetic'.",
    )
    synthetic.add_argument("--output-root", type=str, default="workload_output")

    real = subparsers.add_parser(
        "real",
        help="Sample real datasets from parquet and generate matching workloads.",
    )
    real.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the project experiment config JSON. Defaults to "
            "../experiment_config.json or EXPERIMENT_CONFIG."
        ),
    )
    real.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment key inside the config. Defaults to 'real'.",
    )
    real.add_argument("--parquet-path", type=str, required=True)
    real.add_argument("--output-root", type=str, default="workload_output")
    real.add_argument(
        "--world-grid-path",
        type=str,
        default="count_grid_0p05x0p1_deg.npz",
    )
    real.add_argument("--build-world-grid", action="store_true")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_config = load_project_config(args.config)
    experiment_name = args.experiment or args.mode
    experiment_config = get_experiment_config(project_config, experiment_name)
    default_cfg = GeneratorConfig()

    if args.mode == "synthetic":
        config = GeneratorConfig(
            seed=config_value(experiment_config, "seed", default_cfg.seed),
            synthetic_n_points=config_value(
                experiment_config,
                "n_points",
                default_cfg.synthetic_n_points,
            ),
            n_queries=config_value(
                experiment_config,
                "n_queries",
                default_cfg.n_queries,
            ),
            synthetic_num_datasets=config_value(
                experiment_config,
                "num_datasets",
                default_cfg.synthetic_num_datasets,
            ),
            synthetic_num_clusters=config_value(
                experiment_config,
                "synthetic_num_clusters",
                default_cfg.synthetic_num_clusters,
            ),
            num_query_scales=config_value(
                experiment_config,
                "num_query_scales",
                default_cfg.num_query_scales,
            ),
            num_query_clusters=config_value(
                experiment_config,
                "num_query_clusters",
                default_cfg.num_query_clusters,
            ),
            target_fractions=list(
                config_value(
                    experiment_config,
                    "target_fractions",
                    default_cfg.target_fractions,
                )
            ),
            density_grid_size=config_value(
                experiment_config,
                "density_grid_size",
                default_cfg.density_grid_size,
            ),
            exact_refine_steps=config_value(
                experiment_config,
                "exact_refine_steps",
                default_cfg.exact_refine_steps,
            ),
        )
        generator = SpatialWorkloadGenerator(config, output_root=args.output_root)
        generator.run_synthetic()
        return

    if args.mode == "real":
        config = GeneratorConfig(
            seed=config_value(experiment_config, "seed", default_cfg.seed),
            n_queries=config_value(
                experiment_config,
                "n_queries",
                default_cfg.n_queries,
            ),
            real_num_samples=config_value(
                experiment_config,
                "num_samples",
                default_cfg.real_num_samples,
            ),
            num_query_scales=config_value(
                experiment_config,
                "num_query_scales",
                default_cfg.num_query_scales,
            ),
            num_query_clusters=config_value(
                experiment_config,
                "num_query_clusters",
                default_cfg.num_query_clusters,
            ),
            target_fractions=list(
                config_value(
                    experiment_config,
                    "target_fractions",
                    default_cfg.target_fractions,
                )
            ),
            density_grid_size=config_value(
                experiment_config,
                "density_grid_size",
                default_cfg.density_grid_size,
            ),
            real_target_points=config_value(
                experiment_config,
                "real_target_points",
                default_cfg.real_target_points,
            ),
            approx_count_low=config_value(
                experiment_config,
                "approx_low",
                default_cfg.approx_count_low,
            ),
            approx_count_high=config_value(
                experiment_config,
                "approx_high",
                default_cfg.approx_count_high,
            ),
            real_knn_k=config_value(
                experiment_config,
                "real_knn_k",
                default_cfg.real_knn_k,
            ),
            real_center_candidates=config_value(
                experiment_config,
                "real_center_candidates",
                default_cfg.real_center_candidates,
            ),
            center_grid_size=config_value(
                experiment_config,
                "center_grid_size",
                default_cfg.center_grid_size,
            ),
            exact_refine_steps=config_value(
                experiment_config,
                "exact_refine_steps",
                default_cfg.exact_refine_steps,
            ),
        )
        generator = SpatialWorkloadGenerator(config, output_root=args.output_root)
        generator.run_real(
            parquet_path=args.parquet_path,
            world_grid_path=args.world_grid_path,
            build_world_grid=args.build_world_grid,
        )
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
