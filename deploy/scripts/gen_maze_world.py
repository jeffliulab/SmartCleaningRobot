"""Generate maze.world SDF for Gazebo, identical to training-side maze.

Uses the exact same DFS algorithm and seed as builders/maze.py to ensure
the Gazebo maze layout matches the Isaac Lab training maze.
"""

import random
import os

ROWS, COLS = 8, 8
CELL_SIZE = 0.45
WALL_H = 0.3
SEED = 42


def generate_dfs_maze(rows, cols, start, rng):
    h = 2 * rows + 1
    w = 2 * cols + 1
    grid = [[True] * w for _ in range(h)]

    sr, sc = start
    stack = [(sr, sc)]
    visited = {(sr, sc)}
    grid[2 * sr + 1][2 * sc + 1] = False

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                neighbors.append((nr, nc, dr, dc))
        if neighbors:
            nr, nc, dr, dc = rng.choice(neighbors)
            grid[2 * r + 1 + dr][2 * c + 1 + dc] = False
            grid[2 * nr + 1][2 * nc + 1] = False
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()
    return grid


def carve_chamber(grid, room_row, room_col, size_r, size_c):
    for r in range(room_row, room_row + size_r):
        for c in range(room_col, room_col + size_c):
            grid[2 * r + 1][2 * c + 1] = False
            if r > room_row:
                grid[2 * r][2 * c + 1] = False
            if c > room_col:
                grid[2 * r + 1][2 * c] = False


def add_loops(grid, rows, cols, count, rng):
    candidates = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                wr, wc = 2 * r + 1, 2 * (c + 1)
                if grid[wr][wc]:
                    candidates.append((wr, wc))
            if r + 1 < rows:
                wr, wc = 2 * (r + 1), 2 * c + 1
                if grid[wr][wc]:
                    candidates.append((wr, wc))
    rng.shuffle(candidates)
    for wr, wc in candidates[:count]:
        grid[wr][wc] = False


def ensure_center_clear(grid, rows, cols):
    cr, cc = rows // 2, cols // 2
    grid[2 * cr + 1][2 * cc + 1] = False
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = cr + dr, cc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if not grid[2 * nr + 1][2 * nc + 1]:
                grid[2 * cr + 1 + dr][2 * cc + 1 + dc] = False
                break


def main():
    rng = random.Random(SEED)

    grid = generate_dfs_maze(ROWS, COLS, start=(ROWS // 2, COLS // 2), rng=rng)
    # Chambers and loops disabled — they create cycles that break the
    # left-hand-rule wall follower.  Re-enable for RL-only mazes.
    # carve_chamber(grid, room_row=1, room_col=1, size_r=2, size_c=2)
    # carve_chamber(grid, room_row=5, room_col=5, size_r=2, size_c=2)
    # add_loops(grid, ROWS, COLS, count=5, rng=rng)

    h = 2 * ROWS + 1
    w = 2 * COLS + 1
    grid[h - 1][2 * (COLS // 2) + 1] = False
    ensure_center_clear(grid, ROWS, COLS)

    # Use center cell as world origin so robot spawns in a passage at (0, 0)
    cx_off = 2 * (ROWS // 2) + 1  # = 9 for 8x8
    cy_off = 2 * (COLS // 2) + 1  # = 9 for 8x8
    walls = []
    idx = 0
    for row in range(h):
        for col in range(w):
            if grid[row][col]:
                wx = (col - cx_off) * CELL_SIZE
                wy = (cy_off - row) * CELL_SIZE
                walls.append((f"mw{idx}", wx, wy))
                idx += 1

    lines = [
        '<?xml version="1.0"?>',
        "<!--",
        "  Maze environment for Gazebo.",
        "  Generated from the same DFS algorithm as training (builders/maze.py).",
        "  8x8 maze, cell_size=0.45m, wall_h=0.3m, seed=42.",
        "  Features: 2 chambers, 5 loops, deep dead-ends, exit at south center.",
        "  No ceiling. Scene bounds: (-4, -4) to (4, 4).",
        "-->",
        '<sdf version="1.6">',
        '  <world name="maze">',
        "",
        "    <include><uri>model://ground_plane</uri></include>",
        "    <include><uri>model://sun</uri></include>",
        "",
        '    <physics type="ode">',
        "      <real_time_factor>1</real_time_factor>",
        "      <max_step_size>0.001</max_step_size>",
        "    </physics>",
        "",
    ]

    wz = WALL_H / 2.0
    for name, wx, wy in walls:
        lines.append(f'    <model name="{name}">')
        lines.append("      <static>true</static>")
        lines.append(f"      <pose>{wx:.4f} {wy:.4f} {wz:.4f} 0 0 0</pose>")
        lines.append('      <link name="link">')
        lines.append(
            f'        <collision name="c"><geometry><box>'
            f"<size>{CELL_SIZE} {CELL_SIZE} {WALL_H}</size>"
            f"</box></geometry></collision>"
        )
        lines.append(
            f'        <visual name="v"><geometry><box>'
            f"<size>{CELL_SIZE} {CELL_SIZE} {WALL_H}</size>"
            f"</box></geometry>"
        )
        lines.append(
            "          <material><ambient>0.55 0.55 0.7 1</ambient></material></visual>"
        )
        lines.append("      </link>")
        lines.append("    </model>")
        lines.append("")

    lines.append("  </world>")
    lines.append("</sdf>")
    lines.append("")

    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "cleaning_robot_ros",
        "worlds",
        "maze.world",
    )
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated {len(walls)} walls -> {out_path}")


if __name__ == "__main__":
    main()
