"""GPU-accelerated object state tracking for batched RL environments.

Tracks per-object states across multiple parallel environments:
    FREE     — object is on the floor, not held
    GRASPED  — object is held by the robot arm
    PLACED   — object has been placed in the basket

Provides basket containment checking and pickup counting utilities.
Similar to CoverageTracker but for discrete object events.
"""

from __future__ import annotations

import torch


class ObjectTracker:
    """Tracks pickup / placement events for N objects across E environments.

    Args:
        num_envs:      Number of parallel environments.
        num_objects:   Number of small objects per environment.
        device:        Torch device.
    """

    FREE: int = 0
    GRASPED: int = 1
    PLACED: int = 2

    def __init__(self, num_envs: int, num_objects: int, device: str | torch.device):
        self.num_envs = num_envs
        self.num_objects = num_objects
        self.device = device

        # (E, M) — integer state for each object
        self.states = torch.zeros(num_envs, num_objects, dtype=torch.long, device=device)

        # (E,) — count of objects in basket per environment
        self.placed_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_free(self) -> torch.Tensor:
        """(E, M) bool: object is on the floor."""
        return self.states == self.FREE

    def is_grasped(self) -> torch.Tensor:
        """(E, M) bool: object is held by the arm."""
        return self.states == self.GRASPED

    def is_placed(self) -> torch.Tensor:
        """(E, M) bool: object is in the basket."""
        return self.states == self.PLACED

    def any_grasped(self) -> torch.Tensor:
        """(E,) bool: at least one object is grasped in each env."""
        return self.is_grasped().any(dim=-1)

    def get_placed_count_norm(self, max_objects: int | None = None) -> torch.Tensor:
        """(E,) float: basket count normalised to [0, 1]."""
        denom = max_objects if max_objects else self.num_objects
        return self.placed_count.float() / denom

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def mark_grasped(self, env_ids: torch.Tensor, obj_ids: torch.Tensor) -> None:
        """Mark specified (env, object) pairs as GRASPED."""
        self.states[env_ids, obj_ids] = self.GRASPED

    def mark_dropped(self, env_ids: torch.Tensor, obj_ids: torch.Tensor) -> None:
        """Return grasped objects to FREE (dropped without placement)."""
        mask = self.states[env_ids, obj_ids] == self.GRASPED
        self.states[env_ids[mask], obj_ids[mask]] = self.FREE

    def mark_placed(self, env_ids: torch.Tensor, obj_ids: torch.Tensor) -> torch.Tensor:
        """Mark objects as PLACED in basket and increment count.

        Returns:
            (len(env_ids),) bool — True for each pair that was newly placed.
        """
        was_grasped = self.states[env_ids, obj_ids] == self.GRASPED
        self.states[env_ids[was_grasped], obj_ids[was_grasped]] = self.PLACED
        for i, (e, was) in enumerate(zip(env_ids.tolist(), was_grasped.tolist())):
            if was:
                self.placed_count[e] += 1
        return was_grasped

    # ------------------------------------------------------------------
    # Basket containment check
    # ------------------------------------------------------------------

    @staticmethod
    def check_in_basket(
        object_positions: torch.Tensor,  # (E, M, 3) world positions
        basket_positions: torch.Tensor,  # (E, 3) basket centre world position
        basket_size: tuple[float, float, float] = (0.18, 0.18, 0.12),
    ) -> torch.Tensor:                   # (E, M) bool
        """Return True for each object currently inside the basket AABB.

        Args:
            object_positions: (E, M, 3) object world positions.
            basket_positions: (E, 3) basket centre-bottom world positions.
            basket_size:      (sx, sy, sz) interior dimensions of basket (m).
        """
        sx, sy, sz = basket_size
        half = torch.tensor([sx / 2, sy / 2, sz / 2], device=object_positions.device)

        # Basket occupies [centre - half, centre + half]
        lo = basket_positions.unsqueeze(1) - half  # (E, 1, 3)
        hi = basket_positions.unsqueeze(1) + half  # (E, 1, 3)

        inside = (object_positions >= lo) & (object_positions <= hi)  # (E, M, 3)
        return inside.all(dim=-1)                                       # (E, M)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | list[int]) -> None:
        """Reset object states and basket counts for the given environments."""
        self.states[env_ids] = self.FREE
        self.placed_count[env_ids] = 0
