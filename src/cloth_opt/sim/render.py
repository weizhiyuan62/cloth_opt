from pathlib import Path
import os
import shutil
import subprocess
import tempfile
from typing import Any

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


class PolyscopeRenderer:
    """Single-camera renderer using the simulator's native Polyscope stack."""

    def __init__(
        self,
        triangles: np.ndarray,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        camera_offset_xz: tuple[float, float] = (0.8, 0.8),
        camera_height: float = 0.8,
        camera_target_height: float = 0.3,
        image_width: int = 1280,
        image_height: int = 720,
        engine: str = "auto",
        material: str = "wax",
        background_color: tuple[float, float, float] | None = None,
        cloth_color: tuple[float, float, float] = (0.3, 0.7, 0.9),
        edge_width: float = 1.0,
        smooth_shade: bool = False,
        ground_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
        ground_margin: float = 0.2,
        ground_height: float = 0.0,
        control_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        control_radius: float = 0.015,
        target_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
        target_radius: float = 0.012,
    ) -> None:
        try:
            import polyscope as ps
        except ImportError as error:
            raise RuntimeError(
                "native rendering requires Polyscope; install the project again "
                "or run: conda install -c conda-forge polyscope"
            ) from error

        if image_width <= 0 or image_height <= 0:
            raise ValueError("native render image dimensions must be positive")
        if edge_width < 0.0 or ground_margin < 0.0:
            raise ValueError("native edge width and ground margin must be non-negative")
        if control_radius <= 0.0 or target_radius <= 0.0:
            raise ValueError("native marker radii must be positive")
        engine_names = {
            "auto": None,
            "egl": "openGL3_egl",
            "glfw": "openGL3_glfw",
        }
        if engine not in engine_names:
            raise ValueError("native render engine must be auto, egl, or glfw")

        self._ps = ps
        self._triangles = np.asarray(triangles, dtype=np.int64)
        self._mesh = None
        self._ground = None
        self._controlled = None
        self._targets = None
        self._closed = False
        self._material = material
        self._camera_offset_xz = np.asarray(camera_offset_xz, dtype=np.float64)
        if self._camera_offset_xz.shape != (2,):
            raise ValueError("native camera.offset_xz must contain two values")
        self._camera_height = float(camera_height)
        self._camera_target_height = float(camera_target_height)
        self._cloth_color = tuple(float(value) for value in cloth_color)
        self._edge_width = float(edge_width)
        self._smooth_shade = bool(smooth_shade)
        self._ground_color = tuple(float(value) for value in ground_color)
        self._ground_margin = float(ground_margin)
        self._ground_height = float(ground_height)
        self._control_color = tuple(float(value) for value in control_color)
        self._control_radius = float(control_radius)
        self._target_color = tuple(float(value) for value in target_color)
        self._target_radius = float(target_radius)
        del bounds  # Native camera/ground geometry is derived from the initial cloth.

        ps.set_allow_headless_backends(True)
        selected_engine = engine_names[engine]
        if selected_engine is None:
            ps.init()
        else:
            ps.init(selected_engine)
        ps.set_up_dir("y_up")
        ps.set_front_dir("z_front")
        ps.set_window_size(int(image_width), int(image_height))
        ps.set_window_resizable(False)
        # The original Optimization app registers its own gray ground mesh.
        ps.set_ground_plane_mode("none")
        if background_color is not None:
            ps.set_background_color(tuple(float(value) for value in background_color))

    def _initialize_structures(
        self,
        positions: np.ndarray,
        controlled_positions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self._mesh = self._ps.register_surface_mesh(
            "Cloth",
            positions,
            self._triangles,
            color=self._cloth_color,
            edge_width=self._edge_width,
            smooth_shade=self._smooth_shade,
            material=self._material,
        )
        minimum = positions[:, [0, 2]].min(axis=0) - self._ground_margin
        maximum = positions[:, [0, 2]].max(axis=0) + self._ground_margin
        ground_vertices = np.asarray(
            [
                [minimum[0], self._ground_height, minimum[1]],
                [maximum[0], self._ground_height, minimum[1]],
                [maximum[0], self._ground_height, maximum[1]],
                [minimum[0], self._ground_height, maximum[1]],
            ],
            dtype=np.float64,
        )
        ground_triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self._ground = self._ps.register_surface_mesh(
            "Ground", ground_vertices, ground_triangles, color=self._ground_color
        )
        self._controlled = self._ps.register_point_cloud(
            "Controlled vertices",
            controlled_positions,
            radius=self._control_radius,
            color=self._control_color,
        )
        self._targets = self._ps.register_point_cloud(
            "Position targets",
            targets,
            radius=self._target_radius,
            color=self._target_color,
        )
        center_xz = 0.5 * (
            positions[:, [0, 2]].min(axis=0) + positions[:, [0, 2]].max(axis=0)
        )
        camera_position = (
            center_xz[0] + self._camera_offset_xz[0],
            self._camera_height,
            center_xz[1] + self._camera_offset_xz[1],
        )
        camera_target = (
            center_xz[0],
            self._camera_target_height,
            center_xz[1],
        )
        self._ps.look_at(camera_position, camera_target)

    def save_frame(
        self,
        positions: np.ndarray,
        controlled_indices: np.ndarray,
        targets: np.ndarray,
        path: str | Path,
        title: str,
    ) -> None:
        del title  # Native screenshots intentionally omit debug UI text.
        if self._closed:
            raise RuntimeError("native renderer is already closed")
        positions = np.asarray(positions, dtype=np.float64)
        controlled_positions = positions[np.asarray(controlled_indices, dtype=np.int64)]
        targets = np.asarray(targets, dtype=np.float64)
        if self._mesh is None:
            self._initialize_structures(positions, controlled_positions, targets)
        else:
            self._mesh.update_vertex_positions(positions)
            self._controlled.update_point_positions(controlled_positions)
            self._targets.update_point_positions(targets)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ps.screenshot(str(path), transparent_bg=False, include_UI=False)

    def close(self) -> None:
        if self._closed:
            return
        self._ps.remove_all_structures()
        shutdown = getattr(self._ps, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self._closed = True


def make_single_camera_renderer(
    triangles: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    render_cfg: Any,
) -> SingleCameraRenderer | PolyscopeRenderer:
    """Construct the configured native or debug renderer."""

    backend = str(render_cfg.backend)
    camera = render_cfg.camera
    if backend == "matplotlib":
        return SingleCameraRenderer(
            triangles,
            bounds,
            elevation=float(camera.elevation),
            azimuth=float(camera.azimuth),
        )
    if backend == "polyscope":
        appearance = render_cfg.appearance
        return PolyscopeRenderer(
            triangles,
            bounds,
            camera_offset_xz=tuple(float(value) for value in camera.offset_xz),
            camera_height=float(camera.height),
            camera_target_height=float(camera.target_height),
            image_width=int(render_cfg.image_width),
            image_height=int(render_cfg.image_height),
            engine=str(render_cfg.engine),
            material=str(render_cfg.material),
            background_color=(
                None
                if appearance.background_color is None
                else tuple(float(value) for value in appearance.background_color)
            ),
            cloth_color=tuple(float(value) for value in appearance.cloth_color),
            edge_width=float(appearance.edge_width),
            smooth_shade=bool(appearance.smooth_shade),
            ground_color=tuple(float(value) for value in appearance.ground_color),
            ground_margin=float(appearance.ground_margin),
            ground_height=float(appearance.ground_height),
            control_color=tuple(float(value) for value in appearance.control_color),
            control_radius=float(appearance.control_radius),
            target_color=tuple(float(value) for value in appearance.target_color),
            target_radius=float(appearance.target_radius),
        )
    raise ValueError(f"unsupported render backend: {backend}")


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
