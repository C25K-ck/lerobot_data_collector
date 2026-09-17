

class FSM():
    def __init__(self):
        pass
        self.task_list = ["task_1_pp", "task_2_pp", "task_3_pp"]
        self.task_list = ["task_1_res34", "task_2_res34",  "task_2_res34", "task_4_res34"]
        # self.task_list = ["task_1_res34", "task_5_res34",  "task_2_res34", "task_2_res34", "task_5_res34"]
        # self.task_list = ["task_1_res34", "task_2_pp", "task_3_res34", "task_4_res34"]
        self.task_detect_list= ["task_1_detect", "task_2_detect", "task_1_detect", "task_1_detect", "task_1_detect"]
        self.task_idx = -1

    def check_complete(self, is_complete):
        if(is_complete):
            pass

    def get_target_qpos(self, current_qpos, qpos):
        pass

