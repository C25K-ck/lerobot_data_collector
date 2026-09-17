from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parents[2])

RIGHT_ARM_URDF = str(Path(BASE_PATH) / "robot_kinemic/robot_kinemic/model/urdfs/r_arm_v2.urdf")

LEFT_ARM_URDF = str(Path(BASE_PATH) / "robot_kinemic/robot_kinemic/model/urdfs/l_arm_v2.urdf")

TeleopData_IP = "192.168.54.100"
TeleopData_IP = "192.168.54.100"
