import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import requests
import matplotlib.pyplot as plt
 
import torch
import torchvision.transforms as T
from torch.nn.functional import linear,softmax
torch.set_grad_enabled(False)
import cv2 
import numpy as np
from skimage.transform import resize
from scipy.ndimage import zoom

conv_features, enc_attn_weights, dec_attn_weights = [], [], []
cq = []     # 存储detr中的 cq
pk =  []    # 存储detr中的 encoder pos
memory = [] # 编码器最后一层的输入/解码器的输入特征

IDX = 0
class visualizeFeature():
    def __init__(self,model) -> None:
        pass
        self.model = model
        self.load_model(self.model)




    def load_model(self,model):
        # ----------------------------------------------1. 加载模型及获取训练好的参数---------------------------------------------------
        # 加载线上的模型
        # model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

        # 获取训练好的参数
        # for k, v in model.state_dict().items():
        #     print(k)

        for name, parameters in model.named_parameters():
            # 获取训练好的object queries，即pq:[100,256]
            if name == 'model.query_embed.weight':
                pq = parameters
            # 获取解码器的最后一层的交叉注意力模块中q和k的线性权重和偏置:[256*3,256]，[768]
            if name == 'model.transformer.decoder.layers.6.multihead_attn.in_proj_weight':
                in_proj_weight = parameters
            if name == 'model.transformer.decoder.layers.6.multihead_attn.in_proj_bias':
                in_proj_bias = parameters



        # print(f"pq : {pq}, in_proj_weight: {in_proj_weight}, in_proj_bias: {in_proj_bias}\n")


        # ------------------------------------------------3. 准备存储前馈该图片时的值---------------------------------------------------
        # use lists to store the outputs via up-values

        
       # 注册hook
        self.hooks = [
            # 获取resnet最后一层特征图
            self.model.model.backbones[0][-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output) 
            ),

            # 获取encoder的图像特征图memory
            self.model.model.transformer.encoder.register_forward_hook(
                lambda self, input, output: memory.append(output)
            ),
            # 获取encoder的最后一层layer的self-attn weights
            self.model.model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            # 获取decoder的最后一层layer中交叉注意力的 weights
            self.model.model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
            # 获取decoder最后一层self-attn的输出cq
            self.model.model.transformer.decoder.layers[-1].norm1.register_forward_hook(
                lambda self, input, output: cq.append(output)
            ),
            # 获取图像特征图的位置编码pk
            self.model.model.backbones[0][-1].register_forward_hook(
                lambda self, input, output: pk.append(output)
            ),
        ]
        self.conv_features =conv_features
        self.enc_attn_weights = enc_attn_weights
        self.dec_attn_weights = dec_attn_weights
        self.cq = cq
        self.pk = pk
        self.memory = memory

        

    def get_net_para(self):
        global IDX

       
        # # propagate through the model
        # outputs = model(img)
        
        # 用完的hook后删除
        # for hook in self.hooks:
        #     hook.remove()
        
        # don't need the list anymore
        conv_feature = self.conv_features[IDX]      # [1,2048,25,34]
        enc_attn_weight = self.enc_attn_weights[IDX][0] # [1,850,850]   : [N,L,S]
        dec_attn_weight = self.dec_attn_weights[IDX][0] # [1,100,850]   : [N,L,S] --> [batch, tgt_len, src_len]
        memory = self.memory[0] # [850,1,256] # 编码器最后一层的输入/解码器的输入特征
        
        
        cq = self.cq[0]    # decoder的self_attn:最后一层输出[100,1,256]
        pk = self.pk[0]    # [1,256,25,34]

        # print(f"para : {conv_features}, {enc_attn_weights}, {dec_attn_weights}, {memory}, {cq}, {pk}\n")
        # print(f"enc_attn_weight : {enc_attn_weight.shape}\n")
        print(f"dec_attn_weight : {dec_attn_weight.shape}\n")
        decoder_features = self.plot(conv_features, dec_attn_weight)
        IDX += 1
        return decoder_features


    def plot(self, conv_feature, dec_attn_weight):
        # ----------------------------------------------------------5. 画图---------------------------------------------------------
        # h, w = conv_feature.shape[-2:]

        # fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(22, 28))  # [11,2]

        # ax = axs.T[0][0]
        # ax.imshow(dec_attn_weights[0, idx].view(h, w))
        # ax.axis('off')
        # ax.set_title(f'query id: {idx.item()}',fontsize = 30)

        # fig.tight_layout()        # 自动调整子图来使其填充整个画布

        # plt.show()


        # print(f"dec_attn_weights[0, idx] : {dec_attn_weight[0, idx][0:300]}")
        # print(f"len :conv_features {len(conv_features)}\n")

        # heatmap = dec_attn_weight.mean(dim=0, keepdim=True).squeeze().cpu().numpy()
        # heatmap0 = dec_attn_weight[0][900:1200].squeeze().cpu().numpy()
        heat_maps = []
        feature_ids = [[900,1200],[600,900],[300,600],[000,300]]
        for i in range(4):


            heatmap0 = dec_attn_weight[0][feature_ids[i][0]:feature_ids[i][1]].squeeze().cpu().numpy()
            heatmap0 = cv2.resize(heatmap0, (640, 480))  # 将热力图的大小调整为与原始图像相同
            scale_factor = 32
            heatmap0 = np.reshape(heatmap0, (15,20))
            heatmap1 = zoom(heatmap0, (scale_factor, scale_factor), order=1)

            heatmap2 = np.uint8(normalize_matrix(heatmap1))
            # heatmap2 = np.uint8(255 * heatmap1)  # 将热力图转换为RGB格式
            heatmap = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
            heat_maps.append(heatmap)


        cv2.imshow('Feature Map', heat_maps[3])
        cv2.waitKey(1)

        # heatmap = temp_feature.mean(dim=1, keepdim=True).squeeze().cpu().numpy()
        # heatmap = cv2.resize(heatmap, (640, 480))  # 将热力图的大小调整为与原始图像相同
        # heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        # heatmaps[cam_id] = heatmap


        return heat_maps

        # cv2.destroyAllWindows()
        

def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) * (255 / (max_val - min_val))
    return normalized_matrix



if __name__ =="__main__":
    s = visualizeFeature()
    s.load_model()