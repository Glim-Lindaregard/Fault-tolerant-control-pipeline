from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    controller_node = Node(
        package='slider_ftc_control',
        executable='control_node',   # your controller
        name='controller_node',
        output='screen',
        parameters=[{
            "bounds_mode": "ams",  # box | ellipsoid | ams
        }]
    )

    #allocator_node = Node(
    #    package='slider_ftc_control',
    #    executable='allocator_node',
    #    name='allocator_node',
    #    output='screen'
    #)

    allocator_node = Node(
        package='slider_low_level_controller',
        executable='thrust_to_pwm',
        name='allocator_node',
        output='screen'
    )

    thruster_state_node = Node(
        package='slider_ftc_control',
        executable='thruster_state',
        name='thruster_state_node',
        output='screen',
        parameters=[{
            "start_health": [1, 1, 1, 1, 1, 1, 1, 1],
            "failure_times": [15.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            "failure_states": [2, 1, 1, 1, 1, 1, 1, 1],
        }]
)

    


    return LaunchDescription([
        controller_node,
        allocator_node,
        thruster_state_node,
    ])