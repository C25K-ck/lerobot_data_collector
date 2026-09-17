import os

from utils import *
from constant import *
from policy import ACTDetectPolicy
import pickle





class loadSateteDetect():
    def __init__(self) -> None:
        pass
        self.policy = None


    def load_conf(self, task_name):
        task_config = TASK_CONFIGS[task_name]
        episode_len = task_config['episode_len']
        camera_names = task_config['camera_names']
        ckpt_dir = task_config['ckpt_dir']
        model_name = task_config['model_name']
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
            'model_name': model_name,
            'episode_len': episode_len,
            'state_dim': state_dim,
            'policy_config': policy_config,
            'task_name': task_name,
            'camera_names': camera_names,
            
        }
        return config
    



    def load_model(self, config):

        ckpt_path = os.path.join(config["ckpt_dir"], config['model_name'])
        if(self.policy is None):
            self.policy = ACTDetectPolicy(config["policy_config"])
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda:0')))
        try:                                   
            loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda:0')))
        except:
            loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda:1')))
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

        self.camera_names = config["policy_config"]["camera_names"]
        



    def eval_bc(self):
        while True:

            with torch.inference_mode():
                data = next(self.data_unit.iteration)

                qpos = self.pre_process(data[0])
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                ori_image =data[1:]

                curr_images = get_image(data[1:], self.camera_names)

                detect_state = self.policy(qpos, curr_images)
                print(detect_state)

                if(detect_state==0):
                    text_val = "not put in"
                    color  = (255, 0, 0)
                else:
                    text_val = "put in"
                    color  = (0,255, 0)
 

                for i in range(len(ori_image)):
                
                    # heatmap = heat_maps[i]

                    curr_image = cv2.cvtColor(ori_image[i], cv2.COLOR_RGB2BGR)/ 255.0 * 1


                    superimposed_img = curr_image 
                    cv2.putText(superimposed_img, text_val, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                    # superimposed_img = data[i+1]  # 这里的0.4是热力图强度因子
                    
                    cv2.imshow(self.camera_names[i], superimposed_img)
                    
                    cv2.waitKey(10)




if __name__ =="__main__":

    main()








