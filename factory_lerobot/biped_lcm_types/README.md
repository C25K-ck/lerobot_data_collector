# biped_lcm_types

双足的LCM库


## 关于维度的解释

足底力 12 包括(F<sub>l</sub> M<sub>l</sub> F<sub>r</sub> M<sub>r</sub>)


关节参数维度可以大于12， 包括除腿部外其他关节

contact仍设置为4维， 后面给双臂预留


## 腾空状态数据
使用state_estimator_lcmt.lcm中 isLiftedUp 数据
目前有四种状态：  0: 机器人在地面 1： 机器悬挂在空中  2：机器向上腾空期（跳跃）3：机器向下落体（跳跃） 