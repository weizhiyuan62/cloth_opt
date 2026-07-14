from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import numpy as np


class SingleCameraRenderer:
    """Fixed-view matplotlib renderer for reproducible headless rollouts."""

    def __init__(
        self,
        triangles: np.ndarray,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        elevation: float = 25.0,
        azimuth: float = -45.0,
    ) -> None:
        cache_dir = Path(tempfile.gettempdir()) / "cloth_opt_matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self._triangles = np.asarray(triangles, dtype=np.int64)
        self._figure = plt.figure(figsize=(7, 6), dpi=120)
        self._axis = self._figure.add_subplot(111, projection="3d")
        self._surface = Poly3DCollection([], facecolor="#4fa3d9", edgecolor="#1b4f72", linewidth=0.35)
        self._surface.set_alpha(0.9)
        self._axis.add_collection3d(self._surface)
        self._controlled = self._axis.scatter([], [], [], color="red", s=20, label="controlled vertices")
        self._targets = self._axis.scatter([], [], [], color="limegreen", s=16, label="position targets")

        self._axis.set_xlim(*bounds[0])
        self._axis.set_ylim(*bounds[1])
        self._axis.set_zlim(*bounds[2])
        self._axis.set_xlabel("X")
        self._axis.set_ylabel("Z")
        self._axis.set_zlabel("Y")
        self._axis.set_box_aspect((1.0, 1.0, 0.7))
        self._axis.view_init(elev=elevation, azim=azimuth)
        self._axis.legend(loc="upper right")
        self._figure.tight_layout()

    @staticmethod
    def _plot_coordinates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The simulator is Y-up; matplotlib displays X/Z on the table plane.
        return values[:, 0], values[:, 2], values[:, 1]

    def save_frame(
        self,
        positions: np.ndarray,
        controlled_indices: np.ndarray,
        targets: np.ndarray,
        path: str | Path,
        title: str,
    ) -> None:
        positions = np.asarray(positions)
        plot_positions = positions[:, [0, 2, 1]]
        self._surface.set_verts(plot_positions[self._triangles])
        self._controlled._offsets3d = self._plot_coordinates(positions[controlled_indices])
        self._targets._offsets3d = self._plot_coordinates(np.asarray(targets))
        self._axis.set_title(title)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(path)

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self._figure)


def frames_to_video(frame_dir: str | Path, output_path: str | Path, fps: int = 30) -> None:
    """Encode ``frame_%05d.png`` files with the ffmpeg from the Conda env."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found; install it with: conda install -c conda-forge ffmpeg")
    frame_dir = Path(frame_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )
