"""Scene configurations for Isaac Sim built-in environments.

Each scene config defines:
    - USD path (relative to ISAAC_NUCLEUS_DIR)
    - Robot spawn position within the scene
    - Scene bounds for coverage tracking
    - Recommended env_spacing and max_envs for RL training

Available scenes:
    - Office: Multi-room office (path unverified -- may need adjustment)
    - Hospital: Multi-room hospital (verified in Isaac Lab tutorials)
    - Warehouse: Simple warehouse (verified in Isaac Lab tutorials)

Verified USD paths found in Isaac Lab source code:
    - Hospital:  {ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd
    - Warehouse: {ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd

Note: If a scene fails to load, check the Nucleus browser in Isaac Sim to
find the correct path for your Isaac Sim version.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SceneConfig:
    """Configuration for a simulation scene."""

    name: str
    usd_path: str
    robot_init_pos: tuple[float, float, float]
    robot_init_rot: tuple[float, float, float, float]
    scene_bounds: tuple[float, float, float, float]
    env_spacing: float
    recommended_max_envs: int
    description: str
    fixed_spawn_pos: tuple[float, float] | None = None


OFFICE_SCENE = SceneConfig(
    name="office",
    # WARNING: This path is NOT verified in Isaac Lab source code.
    # If loading fails, browse Nucleus in Isaac Sim to find the correct path,
    # or use "hospital" or "warehouse" scenes which are verified.
    usd_path="/Environments/Office/office.usd",
    robot_init_pos=(0.0, 0.0, 0.05),
    robot_init_rot=(1.0, 0.0, 0.0, 0.0),
    scene_bounds=(-10.0, -10.0, 10.0, 10.0),
    env_spacing=25.0,
    recommended_max_envs=64,
    description="Multi-room office with open floor areas and furniture",
)

HOSPITAL_SCENE = SceneConfig(
    name="hospital",
    usd_path="/Environments/Hospital/hospital.usd",
    robot_init_pos=(0.0, 0.0, 0.05),
    robot_init_rot=(1.0, 0.0, 0.0, 0.0),
    scene_bounds=(-15.0, -15.0, 15.0, 15.0),
    env_spacing=35.0,
    recommended_max_envs=32,
    description="Multi-room hospital with corridors and open spaces (verified)",
)

WAREHOUSE_SCENE = SceneConfig(
    name="warehouse",
    usd_path="/Environments/Simple_Warehouse/warehouse.usd",
    robot_init_pos=(0.0, 0.0, 0.05),
    robot_init_rot=(1.0, 0.0, 0.0, 0.0),
    scene_bounds=(-12.0, -12.0, 12.0, 12.0),
    env_spacing=30.0,
    recommended_max_envs=64,
    description="Simple warehouse with shelves and open floor (verified)",
)

SIMPLE_SCENE = SceneConfig(
    name="simple",
    usd_path="",
    robot_init_pos=(0.0, 0.0, 0.05),
    robot_init_rot=(1.0, 0.0, 0.0, 0.0),
    scene_bounds=(-4.0, -4.0, 4.0, 4.0),
    env_spacing=15.0,
    recommended_max_envs=256,
    description="Simple walled room with two obstacles (procedural, no USD download)",
)

MAZE_SCENE = SceneConfig(
    name="maze",
    usd_path="",
    robot_init_pos=(0.0, 0.0, 0.05),
    robot_init_rot=(1.0, 0.0, 0.0, 0.0),
    scene_bounds=(-4.0, -4.0, 4.0, 4.0),
    env_spacing=15.0,
    recommended_max_envs=128,
    description=(
        "8x8 DFS simply-connected maze with dead-ends, "
        "single exit at south wall (procedural, no ceiling)"
    ),
    fixed_spawn_pos=(0.0, 0.0),
)

AVAILABLE_SCENES: dict[str, SceneConfig] = {
    "simple": SIMPLE_SCENE,
    "maze": MAZE_SCENE,
    "office": OFFICE_SCENE,
    "hospital": HOSPITAL_SCENE,
    "warehouse": WAREHOUSE_SCENE,
}
