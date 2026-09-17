import os
import copy
import sys
sys.path.append("/home/dreame/test/factory_lerobot/act_eval")

import rclpy
from utils import *
from constant import *
from fsm import FSM
import pickle
from ros_act_eval import nodeActEval
import time
IS_REAL = True
if(IS_REAL):
    from lcm_unit import lcmUnit
    from p2p_motion import P2PMotion
from load_model import loadModel

#from keyboard_util import keyboardManage
from std_msgs.msg import Float32MultiArray, Int32
# #TASK_54
INIT_QPOS =[0.09947644127289038, 0.19895288254578075, 1.0146596668902492, -0.30837696572248846, -1.0246073013824941, 0.29842933123024346, 0.09947644127289038, 0.4856112735121662, -0.05237020681982772, -1.9026602887358341, 1.7839480769571834, 2.8874887633698147, -0.22322837615517305, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07557603709292428, 0.0, 0.0, 0.33821989884551285]


INIT_QPOS[26] = 0.0
INIT_QPOS[27] = 0.0
INIT_QPOS[10]+= 0.3

grip_bias = [0,0.,0,0.,0.1, 0.1]
grip_bias_idx = 0


class stateChange():
    def __init__(self):
        self.complete_count = 0
        self.is_ready = False
        self.is_complete = True
        self.idle_step = 0


    def update_complete(self, complete_type, max_step):
        if(self.is_ready):
            self.idle_step -=1
            if(self.idle_step<=0):
                self.is_complete = True
                self.is_ready = False
                self.idle_step = 1
            return

        if(complete_type==1):
            self.complete_count += 1
        else:
            self.complete_count = 0
        if(self.complete_count==max_step):
            self.is_ready = True
            self.complete_count = 0

    def reset(self):
        self.is_complete = True
        self.is_ready = False
        self.idle_step = 1

class actEval(nodeActEval,loadModel, FSM):
    def __init__(self) -> None:
        nodeActEval.__init__(self)
        loadModel.__init__(self)
        FSM.__init__(self)
        self.state_change_m = stateChange()
        if(IS_REAL):
            self.lcm_unit = lcmUnit()
            self.p2p_m = P2PMotion(self.lcm_unit)

        self.time = time.time()

        self.gap_time = time.time() 

        self.t = 0
        self.dim = 30
        self.qpos_numpy = np.array(INIT_QPOS)
        self.pre_qpos_np = copy.deepcopy(self.qpos_numpy)
        self.waist_val = [0.0]
        self.waist_val_pre = [0.0]

        self.env_state = None
        self.is_infer = False

        #self.keyboard_m = keyboardManage()

        self.qpos_cache = []

        self.start_p2p()
        self.check_complete(False)


    def start_p2p(self):

        if(IS_REAL):
            self.p2p_m.start_p2p_move()

            while(not self.p2p_m.is_finish()):
                mocap_data = self.p2p_m.update_qpos(copy.deepcopy(INIT_QPOS))
                self.lcm_unit.send_to_robot(np.array(mocap_data),[mocap_data[26]])

    def check_complete(self, is_complete):
        if(self.task_idx==0):
            self.state_change_m.update_complete(is_complete,40)
        elif(self.task_idx==1):
            self.state_change_m.update_complete(is_complete,40)
        elif(self.task_idx==3):
            self.state_change_m.update_complete(is_complete,40)
        #self.update_keyboard()
        if(self.state_change_m.is_complete):
            # if(self.task_idx==1):
            #     self.control_glue_machine()

            self.task_idx += 1
            self.task_idx %= 1


            self.is_infer = False

            if(self.task_idx==2):
                self.control_glue_machine()

            if(self.task_idx!=2):
                self.load_task()

                # if(self.task_idx==0):
                #     global grip_bias, grip_bias_idx
                #     self.waist_val[0] += grip_bias[grip_bias_idx]
                #     grip_bias_idx +=1 
                #     grip_bias_idx %= len(grip_bias)


                target_qpos = copy.deepcopy(self.init_qpos)

                self.get_target_qpos(self.qpos_numpy, target_qpos)

            # self.load_task()
            # self.get_target_qpos(self.qpos_numpy, self.init_qpos)

            if(self.task_idx==2):
                # self.push_btn_m.idx = 0
                self.replay_action()
                self.task_idx += 1
                self.task_idx %= 1

                self.is_infer = False
                self.load_task()
                self.get_target_qpos(self.qpos_numpy, self.init_qpos)

            # waist bias


            # if(self.task_idx==0):
            #     self.init_qpos[13] -= 0.02


            # if(self.task_idx==1):
            #     self.init_qpos[10] += 0.05
            #     self.init_qpos[9] -= 0.05

            self.data_flag.camera_flag = False
            self.data_flag.camera_waist_flag = False
            
            self.is_infer = True
            self.state_change_m.is_complete = False
            return True
        return False

    def control_glue_machine(self):
        value =  Int32(data=1)
        self.publisher_control_glue_mashine.publish(value)
        print(f"control glue pub data")

        
        time.sleep(5)

    def update_keyboard(self):
        return
        t = time.time()
        if(t - self.gap_time<2):
            return
        if(self.keyboard_m.status[0]== True):
        # if(self.t >=500):
            self.state_change_m.reset()
            self.gap_time = time.time()
        # elif(self.keyboard_m.status[1]== True):
        #     self.task_idx -= 2
        #     self.state_change_m.reset()
        #     self.gap_time = time.time()


    def load_task(self):
        self.init_model(self.task_list[self.task_idx])

        self.t = 0
        self.interplate_ratio = 0.0



    def get_target_qpos(self, current_qpos, qpos):
        # if(self.task_idx==0):
        if(self.task_idx==1 ):
        #if(False):
            gap_step = 400
            ratio_bias = 0.00015
            gap_step =  550
            ratio_bias = 0.00007
        else:
            gap_step = 400
            ratio_bias = 0.00015


        ratio = 0.
        while gap_step>0:
            target_qpos = [0 for _ in range(self.dim)]
            for i in range(self.dim):
                target_qpos[i] = (1-ratio) * current_qpos[i] + ratio * qpos[i]

            target_waist = [0. for _ in range(len(self.waist_val))]
            for i in range(len(self.waist_val)):
                target_waist[i] = (1-ratio) * self.waist_val_pre[i] + ratio * self.waist_val[i]

            self.pub_act_qpos(target_qpos)
            if IS_REAL:
                self.lcm_unit.send_to_robot(np.array(target_qpos), target_waist)

            ratio += ratio_bias
            ratio = min(1, ratio)

            gap_step -= 1
            time.sleep(0.01)
            print(f"gap : {gap_step}")
            current_qpos = copy.deepcopy(target_qpos)

            self.waist_val_pre = copy.deepcopy(target_waist)
        self.qpos_numpy = np.array(current_qpos)
        self.pre_qpos_np = np.array(current_qpos)




    def eval_bc(self):

        task_m =  Int32(data=self.task_idx)
        self.publisher_task_mode.publish(task_m)

        if(not self.is_infer):
            return


        with torch.inference_mode():
                
            # if(not self.data_flag.camera_flag or not self.data_flag.camera_waist_flag):    
                
            if(not self.data_flag.camera_flag ):
                print("camera not finish")
                return
            print(f"task : {self.task_idx}, step : {self.t}\n")
            qpos = self.pre_process(self.qpos_numpy)
            # print(f"qpos : {qpos.shape}")
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            # t1 = time.time()
            # curr_image = get_image(self.data_info.image_list, self.camera_names)
            # t2 = time.time()
            # print(f"image : {t2-t1}")


            all_actions, state_detect = self.policy(qpos, self.curr_image)

            state_detect = state_detect.cpu().numpy()
            beta = self.check_complete(state_detect)
            if(beta):
                return 
            print(f"state_detect : {state_detect}")

            self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = all_actions
            actions_for_curr_step = self.all_time_actions[:, self.t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = self.weight_k
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

            raw_action = raw_action.squeeze(0).cpu().numpy()

            action = self.post_process(raw_action)

            # if(self.task_idx==0):
            #     if(action[20]>60):
            #         action[20] = 80
            #     elif(action[20]<40):
            #         action[20] = 0
                # action[9] -= 0.006

            print(action[7:14])
            target_qpos = self.interpolate_v2(self.pre_qpos_np, action, self.interplate_ratio)

            waist_val = self.interpolate_v2(self.waist_val_pre, self.waist_val, self.interplate_ratio)

            max_ratio = 1.0
            if(self.interplate_ratio<max_ratio):
                # self.interplate_ratio+=0.005
                # self.interplate_ratio+=0.001
                if(self.task_idx==0):
                    self.interplate_ratio+=0.002
                else:
                    self.interplate_ratio += 0.005
                self.interplate_ratio = min(max_ratio, self.interplate_ratio)
                
            self.pre_qpos_np = target_qpos


            self.pub_act_qpos(target_qpos)

            #if(self.keyboard_m.status[1]== True):
            #    target_qpos[20] = 80

            if(self.task_idx==0):
                # target_qpos[20] *= 2


                if(target_qpos[20]<=0.1):
                    target_qpos[20] = 0
                # target_qpos[20] = 1
                # target_qpos[9] -= 0.015

                # print(f"grip : {target_qpos[20]}")

                # target_qpos[9] -= abs(target_qpos[9]+1.5708)*0.05
                # target_qpos[9] -= 0.01
                # target_qpos[10] += 0.02
                pass


            elif(self.task_idx==1):

                # target_qpos[12] -= 0.01           #guanjianaa

                # if(target_qpos[20]<60):
                #     target_qpos[20] = 0
                # if(target_qpos[20]<40):
                #     target_qpos[20] = 0


                # if(self.t%2==0 ):
                #     self.qpos_cache.append(copy.deepcopy(target_qpos))
                pass

            elif(self.task_idx==2):
                return
            
            elif(self.task_idx==3):
                # target_qpos[13] += 0.2

                pass

            
            # elif(self.task_idx==2):
            #     act_val = next(self.push_btn_m.iter)
            #     self.lcm_unit.send_to_robot(np.array(act_val), self.push_btn_m.waist_val)


            #     return
            #     # target_qpos[10] += 0.07
            #     # target_qpos[12] -= 0.02



            # if(self.task_idx==1):
            #     print(target_qpos)
            #     import sys
            #     sys.exit()

            # print(f"target_qpos :{target_qpos}\n")
            # print(f"gripper {target_qpos[20]}\n")

            if IS_REAL:
                self.lcm_unit.send_to_robot(target_qpos, waist_val)


            self.qpos_numpy = copy.deepcopy(target_qpos)


            self.waist_val_pre = copy.deepcopy(waist_val)

            self.t += self.t_gap
            self.data_flag.camera_flag = False
            self.data_flag.camera_waist_flag = False


            # print(time.time()-self.time)
            self.time = time.time()

            # self.check_complete(self.t >=self.max_timesteps)
            if(self.t >=self.max_timesteps):
                self.t = 0





    @staticmethod
    def interpolate_v2(pre_qpos, target_qpos, ratio):
        step_qpos = [0 for _ in range(len(pre_qpos))]
        for i in range(len(pre_qpos)):
            step_qpos[i] = target_qpos[i] * ratio + pre_qpos[i] * (1 - ratio)
        return step_qpos

    def replay_action(self):

        while(len(self.qpos_cache)>0):
            target_qpos = self.qpos_cache.pop()

            target_qpos[10] -= 0.02
            target_qpos[13] -= 0.04
            target_qpos[12] -= 0.065
            # target_qpos[9] += 0.03
            self.pre_qpos_np = copy.deepcopy(target_qpos)
            self.qpos_numpy = copy.deepcopy(target_qpos)
            self.pub_act_qpos(target_qpos)
            if IS_REAL:
                self.lcm_unit.send_to_robot(np.array(target_qpos), self.waist_val)

            time.sleep(0.08)

        self.qpos_cache.clear()







def main():
    rclpy.init()
    act_eval_ros = actEval()

        # 运行节点
    rclpy.spin(act_eval_ros)
    #
    # # 销毁节点，退出ROS2
    act_eval_ros.destroy_node()
    rclpy.shutdown()


if __name__ =="__main__":
    main()
