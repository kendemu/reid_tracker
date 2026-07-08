from glob import glob
from setuptools import find_packages, setup

package_name = 'reid_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models',
            glob('models/*.onnx*') + glob('models/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kendemu',
    maintainer_email='k.demura@qibitech.com',
    description='ReID tracker with RT-DETR, ByteTrack, DINOv3, LightGlue (all-ONNX)',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'reid_node = reid_tracker.reid_node:main',
            'bytetrack_publisher = reid_tracker.bytetrack_publisher:main',
        ],
    },
)
