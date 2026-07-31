from setuptools import setup

package_name = "cr_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jeff",
    maintainer_email="jeff@example.com",
    description="Cleaning Robot S1 学员包：cmd_vel 基础运动控制",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "square_drive = cr_teleop.square_drive:main",
            "scan_inspector = cr_teleop.scan_inspector:main",
            "bag_odom_report = cr_teleop.bag_odom_report:main",
        ],
    },
)
