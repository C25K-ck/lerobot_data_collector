

DATA_DIR = '/home/dreame/model/'
CKPT_DIR = ""
TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },

    'task_2': {
        'ckpt_dir': DATA_DIR + '/task_2',
        'num_episodes': 100,
        'episode_len': 1000,
        'camera_names': ['head', "left_hand","right_hand"]
    },


    'task_3': {
        'ckpt_dir': DATA_DIR + '/task_3',
        'num_episodes': 100,
        'episode_len': 1000,
        'camera_names': ['head', "left_hand","right_hand"]
    },

    'task_4': {
        'ckpt_dir': DATA_DIR + '/task_4',
        'num_episodes': 50,
        'episode_len': 1000,
        'camera_names': ['head', "right_hand"]
    },
        'task_33': {
        'ckpt_dir': DATA_DIR + '/task_33',
        'num_episodes': 30,
        'episode_len': 700,
        'camera_names': ['headf', "right_hand"]
    },
    'task_38_100': {
        'ckpt_dir': DATA_DIR + '/task_38_100',
        'num_episodes': 100,
        'episode_len': 4000,
        'camera_names': ['headf', "right_hand"]
    },

    'task_55': {
        'ckpt_dir': DATA_DIR + '/task_55_100',
        'num_episodes': 100,
        'episode_len': 4000,
        'camera_names': ['headf',"right_hand"]
    },

}