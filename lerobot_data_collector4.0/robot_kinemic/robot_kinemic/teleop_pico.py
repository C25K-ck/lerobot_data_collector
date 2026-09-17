
from pico_stream import PicoxrControllerStreamer

import copy

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from kinemic_utils import *
from conf import *




def buildStreamer(_ip, _record):

    return PicoxrControllerStreamer(ip=_ip, port=12345, record=_record)

server_ip = "192.168.12.110"

class TeleopData():
    def __init__(self, server_ip):

        self.stream_contrller = buildStreamer(server_ip, False)
        self.mocap_map =  {"l_h":{"p_e": [0,0,0,0,0,0], "btn":[0,0,0,0], "r_axis":[0,0]},"r_h":{"p_e": [0,0,0,0,0,0], "btn":[0,0,0,0], "r_axis":[0,0]}, "hmd":{"p_e": [0,0,0,0,0,0]}}


    def get_controller_data_l(self):
        info = self.stream_contrller.latest
        if(not info["left_connect"]):
            return None
        wrist_trans_l = info["left_controller_matrix"]
        pos_eul = np.zeros(6, dtype=float)
        pos_eul[:3] = wrist_trans_l[:3, 3]
        pos_eul[3:] = rot_to_eul(wrist_trans_l[:3,:3])
        btn = info["btn_l"]
        r_axis = info["r_axis_l"]
        mocap_info = {"p_e":pos_eul, "btn":btn, "r_axis":r_axis}
        return mocap_info


    def get_controller_data_r(self):
        info = self.stream_contrller.latest
        if(not info["right_connect"]):
            return None
        wrist_trans_r = info["right_controller_matrix"]
        pos_eul = np.zeros(6, dtype=float)
        pos_eul[:3] = wrist_trans_r[:3, 3]
        pos_eul[3:] = rot_to_eul(wrist_trans_r[:3,:3])
        btn = info["btn_r"]
        r_axis = info["r_axis_r"]
        mocap_info = {"p_e":pos_eul, "btn":btn, "r_axis":r_axis}
        return mocap_info


    def get_head_data(self):
        head = self.stream_contrller.latest["head"][0]

        pos_eul = np.zeros(6, dtype=float)
        pos_eul[:3] = head[:3, 3]
        try:
            pos_eul[3:] = rot_to_eul(head[:3,:3])
        except:
            # print(head[:3, :3])
            # print(f"head rotate err")
            # pos_eul[3:] = np.zeros(3)
            pass
        mocap_info = {"p_e":pos_eul}

        return mocap_info
    



    def update_mocap_info(self):
        l_controller_info = self.get_controller_data_l()
        if(l_controller_info is not None):
            self.mocap_map.update({"l_h":l_controller_info})

        r_controller_info = self.get_controller_data_r()
        if(r_controller_info is not None):
            self.mocap_map.update({"r_h":r_controller_info})

        head_info = self.get_head_data()
        self.mocap_map.update({"hmd":head_info})

        return self.mocap_map




if __name__ == "__main__":
    s = TeleopData(TeleopData_IP)

    while True:
        s.update_mocap_info()
