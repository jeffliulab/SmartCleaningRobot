"""E4 作业 starter：回放 rosbag——这趟跑了多远、最快多快？

用法：python3 bag_odom_report.py <bag 目录>（缺省读 ./square_run）
读取 bag 里的 /odom，按相邻位置差累计里程，按相邻位置差/时间差估速度。

你的任务：补全 main 里的两段 TODO。读 bag 的部分已经写好——
rosbag2_py 的样板代码每个项目都长一样，看一眼认识它即可。
"""

import math
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

ODOM_TOPIC = "/odom"


def read_odom(bag_path):
    """从 bag 里读出 /odom 的 (x, y, 时间戳 ns) 序列。"""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    samples = []
    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic != ODOM_TOPIC:
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        p = msg.pose.pose.position
        samples.append((p.x, p.y, ts))
    return samples


def main():
    bag_path = sys.argv[1] if len(sys.argv) > 1 else "square_run"
    samples = read_odom(bag_path)
    if len(samples) < 2:
        print(f"bag 里没有足够的 {ODOM_TOPIC} 数据（{bag_path}）")
        return

    # TODO(1) 总里程：遍历相邻两个样本，用 math.hypot 算每一步的位移并累加。
    #   提示：zip(samples, samples[1:]) 能同时拿到 (上一帧, 这一帧)。

    # TODO(2) 最高速度：同一次遍历里，用 步长 / 时间差 估瞬时速度，记下最大值。
    #   注意时间戳单位是纳秒；小心除零。

    # —— 算完后把结果填进这三行（变量名自己定）——
    print(f"bag: {bag_path}")
    print(f"样本数: {len(samples)}")
    # print(f"总里程: {...:.2f} m")
    # print(f"最高速度: {...:.2f} m/s")


if __name__ == "__main__":
    main()
