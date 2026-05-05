from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'slider_ftc_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name,
            ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'control_node = slider_ftc_control.fault_tolerant_controller:main',
        'allocator_node = slider_ftc_control.fault_tolerant_allocator:main',
        'thruster_state = slider_ftc_control.thruster_state:main',
        'build_ams_cache = slider_ftc_control.build_ams_cache_node:main',
        ],
    },
)