"""Small object detection and handling sub-project.

This module is reserved for the small object avoidance feature:
    - detector.py: Visual detection of small objects (hair clips, cables, etc.)
    - handler.py: Decision policy (suck / avoid / stop-and-reverse)

Future integration:
    The global coverage policy will switch to this local handler
    when a small object is detected, forming a hierarchical control architecture.
"""
