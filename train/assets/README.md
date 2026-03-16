# Assets

Local robot models and scene files for Isaac Lab training.

- `robots/` — Custom robot USD/URDF files (e.g. modified JetBot variants)
- `worlds/` — Custom scene USD files (e.g. procedurally exported environments)

Currently empty. The project loads all assets from the NVIDIA Nucleus server:
- JetBot: `{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd`
- Scenes: `{ISAAC_NUCLEUS_DIR}/Environments/Hospital/`, `Office/`, `Simple_Warehouse/`
- Procedural scenes (simple room, DFS maze) are generated in code at `smartcleaningrobot/scenes/builders/`
