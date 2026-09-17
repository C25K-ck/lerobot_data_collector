import os
import copy
import rclpy
from utils import *
from constant import *
from policy import ACTPolicy
import pickle
from ros_act_eval import nodeActEval
import time
from lcm_unit import lcmUnit
from robot.seven_planner_humanoid import get_pos_list_seven_segment
import math



# INIT_QPOS[20]=80

INIT_QPOS = [0.0, 0.1599999964237213, -1.7999999523162842, -1.2999999523162842, 1.7999999523162842, 0.5299999713897705, 0.0, 0.20717433257281637, -0.26179999113082886, -1.545407127408148, 2.177548840207961, 2.7320978201708748, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7766990291262136, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

class actEval(nodeActEval):
    def __init__(self) -> None:
        nodeActEval.__init__(self)
        self.task_name = "task_17"
        self.lcm_unit = lcmUnit()
        self.data_list_t = None
        self.d_len = 0
        self.q = INIT_QPOS
        self.get_first_flag = False
        self.send_Q_flag = True
        while True:
            if self.get_current_pos() is True:
                break
        self.init_model(self.task_name)

    def get_current_pos(self):
        if self.lcm_unit.update_once_arm is True:
            self.q = self.lcm_unit.current_robot_state.q
            return True
        else:
            return False

    def load_conf(self, task_name):
        
        task_config = TASK_CONFIGS[task_name]
        episode_len = task_config['episode_len']
        camera_names = task_config['camera_names']
        ckpt_dir = task_config['ckpt_dir']

        state_dim = 30
        lr_backbone = 1e-5
        backbone = 'resnet18'

        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
                            'num_queries': task_config['num_episodes'],
                            'lr_backbone': lr_backbone,
                            'backbone': backbone,
                            'enc_layers': enc_layers,
                            'dec_layers': dec_layers,
                            'nheads': nheads,
                            'camera_names': camera_names,
                            }
        config = {
            'ckpt_dir': ckpt_dir,
            'episode_len': episode_len,
            'state_dim': state_dim,
            'policy_config': policy_config,
            'task_name': task_name,
            'camera_names': camera_names,
        }
        return config

    def load_model(self, config):
        ckpt_names = [f'policy_best.ckpt', "policy_epoch_19900_seed_0.ckpt"]
        ckpt_name = ckpt_names[0]

        ckpt_path = os.path.join(config["ckpt_dir"], ckpt_name)
        policy = policy = ACTPolicy(config["policy_config"])
        loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda:0')))
        print(loading_status)
        policy.cuda()
        policy.eval()
        return policy

    def load_state(self, ckpt_dir):
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        return pre_process, post_process




    def init_model(self, task_name):

        config = self.load_conf(task_name)

        self.policy = self.load_model(config)

        self.pre_process, self.post_process = self.load_state(config["ckpt_dir"])


        self.max_timesteps = config["episode_len"]
        self.num_queries = config["policy_config"]["num_queries"]

        state_dim = config["state_dim"]
        self.camera_names = config["policy_config"]["camera_names"]
        
        self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, state_dim]).cuda()

        self.qpos_numpy = np.array(INIT_QPOS)

        self.t = 0


    def reset_pos(self):

        self.pub_act_qpos(self.qpos_numpy)

        if self.send_Q_flag:
            self.lcm_unit.send_to_robot(np.array(self.qpos_numpy))

        num_count = 5
        for k in range(num_count):
            print(num_count-k)
            time.sleep(1)
            
            
    def send_p2p_points(self):
        for i in range(self.d_len):
            self.lcm_unit.send_to_robot(np.array(self.data_list_t[i]))
            time.sleep(0.004)
            
    def plan2first(self, point):
        start = self.lcm_unit.current_robot_state.q
        self.data_list_t, self.d_len = get_pos_list_seven_segment(start, point, 45/180*math.pi)


    def eval_bc(self):

        with torch.inference_mode():
            if(self.t <self.max_timesteps and self.data_flag.camera_flag):
                # if(self.t == 0):
                #     self.reset_pos()
                self.q = self.lcm_unit.current_robot_state.q
                qpos = self.pre_process(self.qpos_numpy)
                # qpos = np.array(self.q)
                # for i in [0,1,2,3,4,5,6, 12,13, 14,15,16,17,18,19,21,22,23,24,25, 26,27,28,29]:
                #     qpos[i] = INIT_QPOS[i]
                # qpos[20] = 0 
                # print(f"qpos: {qpos}")
                # qpos[7] = qpos[7] + 0.05
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                curr_image = get_image(self.data_info.image_list, self.camera_names)

                all_actions = self.policy(qpos, curr_image)

                self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = all_actions
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

                raw_action = raw_action.squeeze(0).cpu().numpy()
                
                # raw_action = all_actions[0][0].squeeze(0).cpu().numpy()
                action = self.post_process(raw_action)
                target_qpos = action
                print(f"pub :{target_qpos.tolist()}\n")
                while self.get_first_flag:
                    time.sleep(1)
                # if self.t == 0:
                #     self.plan2first(target_qpos)
                self.pub_act_qpos(target_qpos)

                if self.send_Q_flag:
                    # target_qpos[20] = 0
                    self.lcm_unit.send_to_robot(np.array(target_qpos))


                self.qpos_numpy = copy.deepcopy(target_qpos)

                self.t += 1
                self.data_flag.camera_flag = False





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








