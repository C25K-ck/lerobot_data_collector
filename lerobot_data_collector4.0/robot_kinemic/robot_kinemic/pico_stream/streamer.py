import grpc
from pico_stream.grpc_msg import * 
from threading import Thread
# from stream.utils.grpc_utils import * 
import time 
import numpy as np 
from kinemic_utils import *

YUP2ZUP = np.array([[[1, 0, 0, 0], 
                    [0, 0, 1, 0], 
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]]], dtype = np.float64)

def process_matrix(message):
    m = np.array([[message.m00, message.m01, message.m02, message.m03],
                    [message.m10, message.m11, message.m12, message.m13],
                    [message.m20, message.m21, message.m22, message.m23],
                    [0, 0, 0, 1]])
    return m




class PicoxrControllerStreamer:

    def __init__(self, ip, port, record = True): 

        # Meta Quest IP 
        self.ip = ip
        self.port = port
        self.record = record 
        self.recording = [] 
        self.latest = None 
        self.axis_transform = YUP2ZUP
        self.start_streaming()

    def start_streaming(self): 

        stream_thread = Thread(target = self.stream)
        stream_thread.start() 
        while self.latest is None: 
            pass 
        print(' == DATA IS FLOWING IN! ==')
        print('Ready to start streaming.') 


    def stream(self): 

        # request_hand = xrtracking_pb2.HandUpdate()

        requst_controller = xrtracking_pb2.ControllerUpdate()
        try:
            with grpc.insecure_channel(f"{self.ip}:{self.port}") as channel:
                stub = xrtracking_pb2_grpc.TrackingServiceStub(channel)
                # responses = stub.StreamHandUpdates(request_hand)
                responses = stub.StreamControllerUpdates(requst_controller)

                for response in responses:

                    transformations = {

                        "head": process_matrix(response.Head),

                        "left_controller_matrix": process_matrix(response.left_controller.matrix),
                        "right_controller_matrix": process_matrix(response.right_controller.matrix),

                        "head": self.axis_transform @  process_matrix(response.Head),
                        
                        "left_connect":response.left_controller.isConnected,
                        "right_connect":response.right_controller.isConnected,

                        "btn_l": [
                            response.left_controller.btn_one.isPressed, 
                            response.left_controller.btn_two.isPressed, 
                            response.left_controller.trigger_index.isPressed,  
                            response.left_controller.trigger_hand.isPressed, 
                            response.left_controller.btn_one.isTouched, 
                            response.left_controller.btn_two.isTouched, 
                            response.left_controller.trigger_index.isTouched, 
                            response.left_controller.trigger_hand.isTouched,
                            ],
                        
                        "btn_r": [
                            response.right_controller.btn_one.isPressed, 
                            response.right_controller.btn_two.isPressed, 
                            response.right_controller.trigger_index.isPressed,  
                            response.right_controller.trigger_hand.isPressed, 
                            response.right_controller.btn_one.isTouched, 
                            response.right_controller.btn_two.isTouched, 
                            response.right_controller.trigger_index.isTouched, 
                            response.right_controller.trigger_hand.isTouched,
                            ],

                        "r_axis_l": [
                            response.left_controller.thumb_stick.vector2.x, 
                            response.left_controller.thumb_stick.vector2.y],
                        "r_axis_r": [
                            response.right_controller.thumb_stick.vector2.x, 
                            response.right_controller.thumb_stick.vector2.y]
                    }



                    if self.record: 
                        self.recording.append(transformations)
                    self.latest = transformations 

        except Exception as e:
            print(f"An error occurred: {e}")
            pass 

    def get_latest(self): 
        return self.latest
        
    def get_recording(self): 
        return self.recording
    

if __name__ == "__main__": 

    streamer = MetaPlayControllerStreamer(ip = '10.29.230.57')
    while True: 

        latest = streamer.get_latest()
        print(latest)