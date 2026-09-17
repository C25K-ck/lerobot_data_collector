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


# INIT_QPOS = [ 1.24410675e-02, -2.39971611e-02,  1.50013662e+00, -1.83325429e+00,
#  -1.70745857e+00, -1.22576949e-06,  2.06133346e-06, -3.81118213e-01,
#   1.32447618e-01, -1.37148176e+00,  1.38292912e+00,  2.46545877e+00,
#  -1.47669107e-05, -7.96941091e-06,  6.83408215e-06,  1.14110296e-06,
#   3.05860849e-06,  6.66661872e-06,  2.43357843e-06,  1.06333822e-05,
#   1.01175246e+01, -5.28018910e-06,  2.01167534e-06,  1.99306752e-06,
#   7.44955598e-06, -3.34471457e-06,  1.57864069e-02, -1.04619560e-05,
#   7.94330128e-06,  6.99263050e-06]

INIT_QPOS = [0 for _ in range(30)]

PI = 3.1415926
INIT_QPOS[2] = 0.5* PI
INIT_QPOS[3] = -0.5* PI
INIT_QPOS[4] = -0.5*PI

INIT_QPOS[9] = -0.5* PI
INIT_QPOS[10] = 0.5* PI
INIT_QPOS[11] = 0.5*PI


class actEval(nodeActEval):
    def __init__(self) -> None:
        nodeActEval.__init__(self)
        self.task_name = "task_33"
        self.lcm_unit = lcmUnit()
        self.q = INIT_QPOS
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
        ckpt_names = [f'policy_best.ckpt', "policy_epoch_23000_seed_0.ckpt"]
        ckpt_name = ckpt_names[1]

        ckpt_path = os.path.join(config["ckpt_dir"], ckpt_name)
        policy = policy = ACTPolicy(config["policy_config"])
        #loading_status = policy.load_state_dict(torch.load(ckpt_path))
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

        if False:
            self.lcm_unit.send_to_robot(np.array(self.qpos_numpy))

        num_count = 5
        for k in range(num_count):
            print(num_count-k)
            time.sleep(1)


    def eval_bc(self):

        with torch.inference_mode():

            if(self.t <self.max_timesteps and self.data_flag.camera_flag):
                # if(self.t == 0):
                #     self.reset_pos()
                
                # qpos = self.pre_process(self.qpos_numpy)
                self.q = self.lcm_unit.current_robot_state.q
                qpos = self.pre_process(np.array(self.q))
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
                action = self.post_process(raw_action)
                target_qpos = action
                print_target_qpos = [num for num in target_qpos.tolist()]
                print(f"pub :{print_target_qpos}\n")
                # self.pub_act_qpos(target_qpos)
                
                print(f"for check")
                # print(f"{print_target_qpos[0:7]}")
                print(f"{print_target_qpos[7:14]}")
                # print(f"{print_target_qpos[14:20]}")
                # print(f"{print_target_qpos[20:26]}")
                # print(f"{print_target_qpos[26:28]}")
                # print(f"{print_target_qpos[28:30]}")

                while True:
                    time.sleep(1)
                    
                if False:
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








