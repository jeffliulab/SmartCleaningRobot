import os
from glob import glob

from setuptools import find_packages, setup

package_name = "cleaning_robot_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="todo",
    maintainer_email="todo@todo.com",
    description="ROS 2 deployment for RL-trained cleaning robot",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_node = cleaning_robot_ros.policy_node:main",
        ],
    },
)
