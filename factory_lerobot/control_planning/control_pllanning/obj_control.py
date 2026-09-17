import random


class objControl():
    def __init__(self) -> None:
        self.direction = 1
        pass

    def get_obj_pos_eul(self):
        x_pos = random.randrange(-50,500)*0.001
        y_pos = random.randrange(-50,50)*0.001
        self.obj_pos = [0.3+x_pos, 0.+y_pos, 1.08]
    
        self.obj_quat = [-0.7071082, 1.0383573e-08, 0.7071055, 1.9900469e-08]


    def move_test_1(self):
        if(self.obj_pos[1]>0.5):
            self.direction = -1
        elif(self.obj_pos[1]<-0.5):
            self.direction = 1


        self.obj_pos[1] += self.direction * 0.001






