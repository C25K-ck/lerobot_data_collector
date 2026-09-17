import pinocchio
import numpy as np

from kinemic_utils import *
from numpy.linalg import norm, solve
from conf import *
# import plot_util
from numpy.linalg import svd

eps = 0.01
IT_MAX = 1000
DT = 5e-1
damp = 1e-12
import copy
PI = 3.1415926

class loadUrdf():
    def __init__(self, is_left=True):
        self.is_left = is_left

        if (is_left):
            self.urdf_file_path = LEFT_ARM_URDF
        else:
            self.urdf_file_path = RIGHT_ARM_URDF
        self.dim = 6
        self.num_link = 7
        self.end_idx = 7
        self.pos = np.zeros(3)
        self.eul = np.zeros(3)
        self.tp = np.zeros(self.dim)

        self.elbow_pos = np.zeros(3)
        self.elbow_eul = np.zeros(3)

        # if(self.is_left):
        #     self.qpos_threshold = [[-2.7576,2.7576], [-0.2618, 2.5656], [-2.6354, 2.6354], [-2.2166,0.2443], [ -2.9147, 2.9147], [-0.9948, 0.9948], [-0.4538, 0.4538]]
        # else:
        #     self.qpos_threshold = [[-2.7576,2.7576], [-2.5656, 0.2618], [-2.6354, 2.6354], [-0.2443, 2.2166], [ -2.9147, 2.9147], [-0.9948, 0.9948], [-0.4538, 0.4538]]

        if(self.is_left):
            self.qpos_threshold = [[-1.8,0.5], [0, 2.5656], [-2., 2.], [-2.2166, -0.], [ -2.9147, 2.9147], [-0.3948, 0.9948], [-0, 0]]
        else:
            self.qpos_threshold = [[-1.8,0.5], [-2.5656, 0], [-2., 2.], [0., 2.2166], [ -2.9147, 2.9147], [-0.9948, 0.3948], [-0, 0]]

        self.deta_h = [0 for _ in range(self.num_link)]

        self.vel = [0 for _ in range(self.num_link)]
        self.acc = [0 for _ in range(self.num_link)]

        self.weight = np.eye(self.num_link)
        self.iter = 1
        self.err_threshold = 0.01
        self.init_tp = np.zeros(self.dim)

        self.qpos = np.zeros(self.num_link)
        self.load_urdf()
        self.iteration = self.inv_kinemic()

    def load_urdf(self):
        self.model = pinocchio.buildModelFromUrdf(self.urdf_file_path)
        self.data = self.model.createData()
        self.data_cond = self.model.createData()

        qpos = pinocchio.neutral(self.model)
        
        if(self.is_left):
            qpos[0:7] = [-0.17754411,  0.63843851,  0.95991734, -1.54389083,-2.10363551,  0.34365676, 0.38870025]
        else:
            qpos[0:7] = [-0.17754411,  -0.63843851,  -0.95991734, 1.54389083,2.10363551,  -0.34365676, -0.38870025]



        self.qpos = copy.deepcopy(qpos)

        print(f"init qpos {self.is_left}: {qpos}\n")

        self.update_ik(qpos)
        self.tp = np.hstack((self.pos, self.eul))

        self.init_tp = np.hstack((self.pos, self.eul))

        print(f"init {self.init_tp }\n")

        self.init_rot = eul_to_rot(self.eul)



    def update_target_pos_diff(self, target_pos_eul_dff):


        if(self.is_left):
            self.tp[0] = -1*target_pos_eul_dff[2] + self.init_tp[0]
            self.tp[1] = -1*target_pos_eul_dff[0] + self.init_tp[1]
            self.tp[2] = -1*target_pos_eul_dff[1] + self.init_tp[2]

        else:
            self.tp[0] = -1*target_pos_eul_dff[2] + self.init_tp[0]
            self.tp[1] = -1*target_pos_eul_dff[0] + self.init_tp[1]
            self.tp[2] = -1*target_pos_eul_dff[1] + self.init_tp[2]


        deta_rot = eul_to_rot(target_pos_eul_dff[3:])
        tp_rot = self.init_rot @ deta_rot
        
        tp_eul = rot_to_eul(tp_rot)
        self.tp[3] =  tp_eul[0]
        self.tp[4] =  tp_eul[1]
        self.tp[5] =  tp_eul[2]

        # if(self.is_left):
        #     # print(target_pos_eul_dff[:3])

        #     print(f"targe diff left : {target_pos_eul_dff}\n")


    def update_ik(self, qpos):
        pinocchio.forwardKinematics(self.model, self.data, qpos)
        self.jacobian = pinocchio.computeJointJacobian(self.model, self.data, qpos, self.end_idx)
        # print(self.jacobian)
        # self.jacobian = pinocchio.computeFrameJacobian(self.model, self.data, qpos, self.end_idx)

        self.pos = self.data.oMi[self.end_idx].translation
        self.eul = rot_to_eul(self.data.oMi[self.end_idx].rotation)

        # self.elbow_pos = self.data.oMi[4].translation
        # self.elbow_eul = rot_to_eul(self.data.oMi[4].rotation)
        # print(self.jacobian)
        # if(self.is_left):
        #     print(f"left end pos : {self.pos}\n")
        # else:
        #     print(f"right end pos : {self.pos}\n")
        # # self.is_collapse()

    def get_fk_J(self, qpos):
        pinocchio.forwardKinematics(self.model, self.data_cond, qpos)
        J = pinocchio.computeJointJacobian(self.model, self.data_cond, qpos, self.end_idx)
        pos = self.data_cond.oMi[self.end_idx].translation
        rot = self.data_cond.oMi[self.end_idx].rotation
        oMdes = pinocchio.SE3(rot, pos)
        iMd = self.data_cond.oMi[self.end_idx].actInv(oMdes)
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        return J


    def is_collapse(self):
        inter_point = []
        for i in range(5):
            inter_point.append([(self.pos[0] - self.elbow_pos[0])*i/5.+self.elbow_pos[0], (self.pos[1] - self.elbow_pos[1])*i/5.+self.elbow_pos[1]])
        
        for point in inter_point:

            a = 0.22
            b = 0.22
            if((point[0]/a)**2 + (point[1]/b)**2 <1):
                print("COLLAPSE!!!!\n")
                print(f"is_left : {self.is_left} : {point}")
                import sys
                sys.exit()    

    def get_deta_h(self):
        t = self.qpos_threshold
        q = self.qpos
        a = 1
        b = 1
        for i in range(self.num_link):
            deta = abs(((t[i][1] - t[i][0]) ** 2 * (2 * q[i] - t[i][0] - t[i][1])) / (
                    4 * ((t[i][0] - q[i]) ** 2) * ((t[i][1] - q[i]) ** 2) + 1e-12))
            if (deta >= self.deta_h[i]):
                self.weight[i, i] = a + deta * b
            else:
                self.weight[i, i] = a

            self.weight[i, i] = min(1e20, self.weight[i, i])
            # self.deta_h[i] = deta

    def limit_qpos(self, theta):
        for i in range(self.num_link):
            if (theta[i] > self.qpos_threshold[i][1]):
                theta[i] = self.qpos_threshold[i][1]
            elif (theta[i] < self.qpos_threshold[i][0]):
                theta[i] = self.qpos_threshold[i][0]
        return theta



    def limit_vel(self, v):
        vel_ratio = 0.2
        acc_ratio = 0.5
        acc_threshold = 500
        max_vel = 100
        v_1 = [0 for _ in range(self.num_link)]
        for i in range(self.num_link):
            v_1[i] = vel_ratio * v[i] + (1 - vel_ratio) * self.vel[i]
            v_1[i] = min(max_vel, v_1[i])
        return np.array(v_1)
    

        acc = [0 for _ in range(self.num_link)]
        for i in range(self.num_link):
            acc[i] = v_1[i] - self.vel[i]
            if (acc[i] >= acc_threshold):
                acc[i] = acc_threshold
            elif (acc[i] < -acc_threshold):
                acc[i] = -acc_threshold

        for i in range(self.num_link):
            self.acc[i] = acc[i]
            self.vel[i] = self.acc[i] + self.vel[i]

        return np.array(self.vel)

    def inv_kinemic(self):
        dim = self.dim
        num_links = self.num_link

        theta = np.zeros([num_links], dtype=np.float32)

        damping = 100
        weight = np.identity(dim)

        ee = np.hstack((self.pos, self.eul))
        its = 0

        while True:

            tp = self.tp

            oMdes = pinocchio.SE3(eul_to_rot(tp[3:]), tp[:3])
            pinocchio.forwardKinematics(self.model, self.data, self.qpos)
            iMd = self.data.oMi[self.end_idx].actInv(oMdes)

            err = pinocchio.log(iMd).vector  # in joint frame
            err[3] = min(0.4, err[3])
            err[4] = min(0.4, err[4])
            err[5] = min(0.4, err[5])
            if norm(err) < eps or its % self.iter == 0:
                yield np.hstack((theta, np.zeros(1)))

            err *= np.array((1, 1, 1, 0.2, 0.2, 0.2))
            # err *= np.array((0, 0, 0, 0.2, 0.2, 0.2))
            # err *= np.array((1, 1, 1, 0, 0., 0.))


            J = pinocchio.computeJointJacobian(self.model, self.data, self.qpos, self.end_idx)  # in joint frame
            # J = pinocchio.computeFrameJacobian(self.model, self.data, self.qpos, self.end_idx)  # in joint frame

            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            
            cond_w = 0.01

            cond_weight = self.get_cond_weight(J,sigma_th=1e-4,cond_th1=10, cond_th2=100)
            
            if(cond_weight!=0):

                conJ = self.cond_gradient(self.qpos) * cond_w
                v_cond = -1*(np.eye(7)-np.linalg.pinv(J)@J)@conJ
                
            else:
                v_cond = np.zeros(7)

            self.get_deta_h()
            lam = damp * self.weight

            for i in range(self.end_idx):
                lam[i,i] += abs(v_cond[i])


            v_err = -(np.linalg.pinv((J.T) @ J + lam)) @ J.T @ err 


            v_output = self.limit_vel(v_err)

            v_output[5]*= 4
            v_output[6]*= 4

            theta = pinocchio.integrate(self.model, self.qpos, v_output * DT)
            theta = self.limit_qpos(theta)


            self.qpos = theta
            self.update_ik(theta)

            its += 1
            # if(self.is_left):
            #     # print( [[self.tp.tolist()], [self.pos[0], self.pos[1], self.pos[2], self.eul[0], self.eul[1], self.eul[2]]])
            #     plot_util.plot_eul_list[0] = self.tp.tolist()
            #     plot_util.plot_eul_list[1] = [self.pos[0], self.pos[1], self.pos[2], self.eul[0], self.eul[1], self.eul[2]]
            # else:
            #     plot_util.plot_eul_list[2] = self.tp.tolist()
            #     plot_util.plot_eul_list[3] = [self.pos[0], self.pos[1], self.pos[2], self.eul[0], self.eul[1], self.eul[2]]    


    def wrist_control(self):
            limit = [[ -2.9147, 2.9147], [-0.9948, 0.9948], [-0.4538, 0.4538]]
            qpos = copy.deepcopy(self.qpos[4:7])

            # qpos[0] = self.eul[1] - qpos[0]
            if(self.is_left):
                qpos[0] =  self.eul[1] + qpos[0] 
                qpos[1] = -self.eul[0] + 1.5708
                qpos[2] = 0
            else:
                qpos[0] =  -self.eul[1] + qpos[0] 
                qpos[1] = -self.eul[0] - 1.5708
                qpos[2] = 0

            for i in range(3):
                qpos[i] = max(limit[i][0], qpos[i])
                qpos[i] = min(limit[i][1], qpos[i])
            return qpos


    def condition_number(self,J):
        _, S, _ = svd(J)
        min_sigma = S[-1] if len(S) > 0 else 0
        if min_sigma < 1e-12:
            return np.inf
        return S[0] / min_sigma

    def cond_gradient(self,q, eps=1e-6):
        grad = np.zeros(7)
        J = self.get_fk_J(q)
        current_cond = self.condition_number(J)
        
        for i in range(7):
            q_perturb = q.copy()
            q_perturb[i] += eps
            J_perturb = self.get_fk_J(q_perturb)
            # J_perturb = pinocchio.computeJointJacobian(self.model, self.data, q_perturb, self.end_idx)

            cond_perturb = self.condition_number(J_perturb)
            
            # 数值梯度
            grad[i] = (cond_perturb - current_cond) / eps

        return grad

    def get_cond_weight(self, J, sigma_th=1e-4, cond_th1=50, cond_th2=100):

        U, S, Vh = svd(J)
        
        min_sigma = S[-1] 
        if min_sigma < sigma_th:
            import sys
            sys.exit()
        
        cond = S[0] / min_sigma if min_sigma > 0 else np.inf
        
        cond_weight = 0
        if cond > cond_th1:
            cond_weight = min((cond-cond_th1)/(cond_th2-cond_th1), 1)
            # print(f"# cond weight: {cond}， {cond_weight}")

        return cond_weight


if __name__ == "__main__":
    s = loadUrdf()
    s.load_urdf()
