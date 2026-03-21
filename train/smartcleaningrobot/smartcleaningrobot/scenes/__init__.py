"""Scene configurations and builders for the cleaning robot project."""

from .scene_cfg import (
    AVAILABLE_SCENES,
    HOSPITAL_SCENE,
    MAZE_SCENE,
    OFFICE_SCENE,
    ROOM_WITH_OBJECTS_SCENE,
    SIMPLE_SCENE,
    WAREHOUSE_SCENE,
    SceneConfig,
)
from .builders import SCENE_BUILDERS, build_scene
