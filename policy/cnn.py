import torch
import torch.nn as nn
import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../../../../')
from network_modules import *
from Utils import *





class DisentangledPoseEncoder(nn.Module):
    def __init__(self, input_dim_trans = 3,input_dim_rot = 6, output_dim=64):
        super(DisentangledPoseEncoder, self).__init__()

        self.encoder_trans = nn.Sequential(
            nn.Linear(input_dim_trans, 128),  # First MLP layer
            nn.LayerNorm(128),  # LayerNorm
            # nn.LeakyReLU(),
            nn.GELU(),
            nn.Linear(128, 128),  # Second MLP layer
            nn.LayerNorm(128),  # LayerNorm
            # nn.ReLU(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)  # Third MLP layer (Projection head)
        )

        self.encoder_rot = nn.Sequential(
            nn.Linear(input_dim_rot, 128),  # First MLP layer
            nn.LayerNorm(128),  # LayerNorm
            # nn.LeakyReLU(),
            nn.GELU(),
            nn.Linear(128, 128),  # Second MLP layer
            nn.LayerNorm(128),  # LayerNorm
            # nn.LeakyReLU(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim), # Third MLP layer (Projection head)
        )

        self.pose_head = nn.Sequential(
              nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128, batch_first=True),
            	  nn.Linear(128, 128),)

    def forward(self, trans, rots):
        """
        Forward pass for encoding poses.
        Args:
        x (Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
        Tensor: Encoded features of shape (batch_size, output_dim).
        """

        trans_feature = self.encoder_trans(trans)
        rots_feature = self.encoder_rot(rots)

        a = torch.cat([trans_feature,rots_feature],dim=1 )


        return self.pose_head(a)


class PoseResGateFusion(nn.Module):
  def __init__(self, pose_dim=128, img_dim=1200, hidden_dim=128):
    super().__init__()
    self.img_compress = nn.Sequential(
      nn.Linear(img_dim, hidden_dim),
      nn.LayerNorm(hidden_dim),
      nn.ReLU()
    )

    self.gate_net = nn.Sequential(
      nn.Linear(pose_dim, hidden_dim),
      nn.Sigmoid()
    )

    # self.pose_enhance = nn.Linear(pose_dim, hidden_dim)

  def forward(self, pose_feat, img_feat):
    img_compressed = self.img_compress(img_feat)  # (B, H)

    gate = self.gate_net(pose_feat)  # (B, H)

    gated_img = gate * img_compressed  # (B, H)

    # pose_enhanced = self.pose_enhance(pose_feat)  # (B, H)
    fused = pose_feat + gated_img  # (B, H)
    return fused



class RefineNet(nn.Module):
  def __init__(self, cfg=None, c_in=4, n_view=1):
    super().__init__()
    self.cfg = cfg
    if self.cfg.use_BN:
      norm_layer = nn.BatchNorm2d
      norm_layer1d = nn.BatchNorm1d
    else:
      norm_layer = None
      norm_layer1d = None

    self.encodeA = nn.Sequential(
      ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
    )

    self.encodeAB = nn.Sequential(
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
    )

    embed_dim = 512
    num_heads = 4
    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)

    self.trans_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, 3),
    )

    if self.cfg['rot_rep']=='axis_angle':
      rot_out_dim = 3
    elif self.cfg['rot_rep']=='6d':
      rot_out_dim = 6
    else:
      raise RuntimeError
    self.rot_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, rot_out_dim),
    )


  def forward(self, A, B):
    """
    @A: (B,C,H,W)
    """
    bs = len(A)
    output = {}

    x = torch.cat([A,B], dim=0)
    x = self.encodeA(x)
    a = x[:bs]
    b = x[bs:]

    ab = torch.cat((a,b),1).contiguous()
    ab = self.encodeAB(ab)  #(B,C,H,W)

    ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))

    # output['trans'] = self.trans_head(ab).mean(dim=1)
    # output['rot'] = self.rot_head(ab).mean(dim=1)

    return ab

# fusioner = DisentangledPoseEncoder()
# print(fusioner)
# trans = torch.randn([40,3])
# rots = torch.randn([40,6])
#
# aaa =fusioner.forward(trans,rots )
#
# print(aaa)