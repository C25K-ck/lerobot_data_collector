import os
import copy
import rclpy
from utils import *
from constant import *
from policy import ACTPolicy
import pickle
from ros_act_eval import nodeActEval
from load_data import dataUnit

from visualize_feature import *

class modelEval(nodeActEval):
    def __init__(self) -> None:
        nodeActEval.__init__(self)
        self.task_name = "task_56"
        self.init_model(self.task_name)
        self.data_unit = dataUnit()


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
        ckpt_names = [f'policy_best.ckpt', "policy_epoch_9000_seed_0.ckpt"]
        ckpt_name = ckpt_names[1]

        ckpt_path = os.path.join(config["ckpt_dir"], ckpt_name)
        policy = policy = ACTPolicy(config["policy_config"])
        loading_status = policy.load_state_dict(torch.load(ckpt_path,  map_location='cuda:0'))
        print(loading_status)
        policy.cuda()
        policy.eval()

        # self.visualize_feature = visualizeFeature(policy)
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





    def eval_bc(self):

        with torch.inference_mode():
            data = next(self.data_unit.iteration)

            qpos = self.pre_process(data[0])
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            ori_image =data[2:]

            curr_images = get_image(data[2:], self.camera_names)

            all_actions, heat_maps = self.policy(qpos, curr_images)

            for i in range(len(ori_image)):
            
                heatmap = heat_maps[i]

                curr_image =    cv2.cvtColor(ori_image[i], cv2.COLOR_RGB2BGR)/ 255.0 * 1


                superimposed_img = curr_image + heatmap*0.01
                print(superimposed_img)
                # superimposed_img = data[i+1]  # 这里的0.4是热力图强度因子
                
                cv2.imshow(self.camera_names[i], superimposed_img)
                cv2.waitKey(10)

            # self.visualize_feature.get_net_para()




def main():
    rclpy.init()
    model_eval_ros = modelEval()

        # 运行节点
    rclpy.spin(model_eval_ros)
    #
    # # 销毁节点，退出ROS2
    model_eval_ros.destroy_node()
    rclpy.shutdown()


if __name__ =="__main__":

    main()








