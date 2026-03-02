"""Simple validation policies for pipeline testing.

V1 algorithms (route_plan, route_follow) are archived on the noetic branch.
This module provides a minimal random exploration policy for verifying that
the robot, sensors, scene, and coverage tracker work end-to-end.
"""

from .random_explorer import RandomExplorerPolicy
