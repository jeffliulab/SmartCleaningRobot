"""Small object asset configurations for Isaac Lab.

Defines RigidObjectCfg templates for the four small object types used in
T6 (CoverageAvoid) and T7 (CoveragePickup) tasks. Objects are procedurally
generated — no external USD files required.

Physical specs (based on real household objects):
    hairpin:  blue metallic capsule, length=50mm, r=3mm, mass=3g
    cable:    red cuboid segment, 120×8×3mm, mass=10g
    coin:     gold cylinder, r=12mm, h=2mm, mass=6g
    eraser:   purple cuboid, 45×20×10mm, mass=15g

Usage::

    from smartcleaningrobot.small_objects.object_assets import make_object_cfg
    from isaaclab.assets import RigidObject

    # Create object slot 2 as a coin
    cfg = make_object_cfg(object_type="coin", slot_index=2)
    obj = RigidObject(cfg)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# ---------------------------------------------------------------------------
# Shared physics properties
# ---------------------------------------------------------------------------

_RIGID = sim_utils.RigidBodyPropertiesCfg(
    max_linear_velocity=2.0,
    max_angular_velocity=50.0,
    max_depenetration_velocity=1.0,
)

_COLLISION = sim_utils.CollisionPropertiesCfg(
    contact_offset=0.001,
    rest_offset=0.0,
)


def _surface(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.1)


def _friction(static: float, dynamic: float) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=static,
        dynamic_friction=dynamic,
        restitution=0.05,
    )


# ---------------------------------------------------------------------------
# Per-type configs (prim_path is a placeholder; use make_object_cfg to fill it)
# ---------------------------------------------------------------------------

HAIRPIN_CFG = RigidObjectCfg(
    prim_path="__placeholder__",
    spawn=sim_utils.MeshCapsuleCfg(
        radius=0.003,
        height=0.050,
        rigid_props=_RIGID,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.003),
        collision_props=_COLLISION,
        physics_material=_friction(0.5, 0.4),
        visual_material=_surface((0.30, 0.50, 0.90)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.003)),
)

CABLE_CFG = RigidObjectCfg(
    prim_path="__placeholder__",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.120, 0.008, 0.003),
        rigid_props=_RIGID,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.010),
        collision_props=_COLLISION,
        physics_material=_friction(0.5, 0.4),
        visual_material=_surface((0.90, 0.20, 0.15)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0015)),
)

COIN_CFG = RigidObjectCfg(
    prim_path="__placeholder__",
    spawn=sim_utils.MeshCylinderCfg(
        radius=0.012,
        height=0.002,
        rigid_props=_RIGID,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.006),
        collision_props=_COLLISION,
        physics_material=_friction(0.30, 0.25),
        visual_material=_surface((0.85, 0.65, 0.15)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.001)),
)

ERASER_CFG = RigidObjectCfg(
    prim_path="__placeholder__",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.045, 0.020, 0.010),
        rigid_props=_RIGID,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
        collision_props=_COLLISION,
        physics_material=_friction(0.6, 0.5),
        visual_material=_surface((0.60, 0.20, 0.70)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.005)),
)

# Maps type name → template config (prim_path must be overridden via make_object_cfg)
SMALL_OBJECT_CFGS: dict[str, RigidObjectCfg] = {
    "hairpin": HAIRPIN_CFG,
    "cable": CABLE_CFG,
    "coin": COIN_CFG,
    "eraser": ERASER_CFG,
}

# Approximate bounding sphere radius (m) — used for spawn clearance checks
OBJECT_RADII: dict[str, float] = {
    "hairpin": 0.026,   # half capsule length
    "cable": 0.061,     # half diagonal of 120×8mm
    "coin": 0.012,      # coin radius
    "eraser": 0.025,    # half diagonal of 45×20mm
}

# Half-height above ground (m) — upper bound for basket containment checks
OBJECT_HALF_HEIGHTS: dict[str, float] = {
    "hairpin": 0.006,
    "cable": 0.0015,
    "coin": 0.001,
    "eraser": 0.005,
}

# Object types that the arm can pick up
PICKABLE_TYPES: frozenset[str] = frozenset({"hairpin", "cable", "coin", "eraser"})


def make_object_cfg(object_type: str, slot_index: int) -> RigidObjectCfg:
    """Return a RigidObjectCfg for one object slot in the scene.

    Args:
        object_type: One of the keys in :data:`SMALL_OBJECT_CFGS`.
        slot_index:  0-based index for the object; determines the prim_path.

    Returns:
        Copy of the template config with the correct
        ``/World/envs/env_.*/Objects/obj_{slot_index}`` prim path.
    """
    base = SMALL_OBJECT_CFGS[object_type]
    return base.replace(prim_path=f"/World/envs/env_.*/Objects/obj_{slot_index}")
