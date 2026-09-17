from setuptools import setup

package_name = 'act_get_data'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         

            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),


    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='coco',
    maintainer_email='coco@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "node_act_get_data = act_get_data.robot_control:main"
        ],
    },
)
