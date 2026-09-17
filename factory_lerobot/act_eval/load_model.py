from constant import *
import pickle
import torch
from policy import ACTPolicy
import copy



class loadModel():
    def __init__(self):
        pass
        self.policy = None



    def load_conf(self, task_name):
        
        task_config = TASK_CONFIGS[task_name]
        episode_len = task_config['episode_len']
        camera_names = task_config['camera_names']
        ckpt_dir = task_config['ckpt_dir']
        model_name = task_config['model_name']
        model_type = task_config['model_type']
        init_qpos = task_config['init_qpos']
        waist_val = task_config['waist_val']
        weight_k = task_config['weight_k']
        t_gap = task_config['t_gap']
        state_dim = 30
        lr_backbone = 1e-5
        # backbone = 'resnet18'
        backbone = task_config['backbone']
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
                            'vq': False,
                            'vq_class': 256,
                            'vq_dim': 64,
                            }
        config = {
            'ckpt_dir': ckpt_dir,
            'model_name': model_name,
            'episode_len': episode_len,
            'state_dim': state_dim,
            'policy_config': policy_config,
            'task_name': task_name,
            'camera_names': camera_names,
            'model_type': model_type,
            'init_qpos': init_qpos,
            'waist_val': waist_val,
            'weight_k':weight_k,
            't_gap':t_gap
            
        }
        return config
    


    def load_model(self, config):
        print(config)

        ckpt_path = os.path.join(config["ckpt_dir"], config['model_name'])
        if(self.policy is None):
            self.policy = ACTPolicy(config["policy_config"])
        # self.policy = ACTPolicy(config["policy_config"])
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda:0')))                               
        # try:                                   
        #     loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda:0')))
        # except:
        #     loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda:1')))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()

    

    def load_state(self, ckpt_dir):
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        return pre_process, post_process



    def init_model(self, task_name):
        config = self.load_conf(task_name)
        self.load_model(config)
        self.pre_process, self.post_process = self.load_state(config["ckpt_dir"])
        self.max_timesteps = config["episode_len"]
        self.num_queries = config["policy_config"]["num_queries"]
        state_dim = config["state_dim"]
        self.camera_names = config["policy_config"]["camera_names"]
        self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, state_dim]).cuda()
        self.init_qpos = copy.deepcopy(config["init_qpos"])
        self.waist_val = copy.deepcopy(config["waist_val"])


        self.weight_k = config["weight_k"]
        self.t_gap = config["t_gap"]






if __name__ =="__main__":
    s = loadModel()
    s.init_model("task_2")