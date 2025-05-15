from policy.diffusion import DiffusionUNetPolicy
from policy.cnn import DisentangledPoseEncoder, PoseResGateFusion
from omegaconf import OmegaConf
from .cnn import RefineNet
from Utils import *


class PoseDP(nn.Module):
    def __init__(
        self,
        num_action = 20,
        input_dim = 9,
        obs_feature_dim = 64,
        action_dim = 9,
        hidden_dim = 512,
        prediction="sample",
    ):
        super().__init__()
        num_obs = 1
        self.pose_encoder = DisentangledPoseEncoder()
        self.action_decoder = DiffusionUNetPolicy(action_dim,horizon= num_action, n_obs_steps=num_obs, obs_feature_dim = obs_feature_dim,prediction=prediction)
        self.readout_embed = nn.Embedding(1, hidden_dim)

    def forward(self, pose, actions = None, batch_size = 24):

        trans = pose[:,0,:3,2]
        rots = pose[:,0,:3,:2]
        rots = rots.reshape(rots.shape[0], rots.shape[1]*rots.shape[2])

        pose_feature = self.pose_encoder(trans, rots)

        if actions is not None:
            loss = self.action_decoder.compute_loss(pose_feature, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(pose_feature)
            return action_pred





class PoseRGBD_DP(nn.Module):
    def __init__(
        self,
        num_action = 20,
        input_dim = 9,
        obs_feature_dim = 64,
        action_dim = 9,
        hidden_dim = 512,
        prediction="sample",
    ):
        super().__init__()
        num_obs = 1
        self.pose_encoder = DisentangledPoseEncoder()

        run_name = "2023-10-28-18-33-37"
        cfg = OmegaConf.load(f'/home/sunh/6D_ws/ActivePose/FoundationPose-main/weights/{run_name}/config.yml')
        self.img_encoder = RefineNet(cfg=cfg, c_in=cfg['c_in']).cuda()
        model_name = 'model_best.pth'
        ckpt_dir = f'/home/sunh/6D_ws/ActivePose/FoundationPose-main/weights/{run_name}/{model_name}'
        ckpt = torch.load(ckpt_dir)
        if 'model' in ckpt:
            ckpt = ckpt['model']
        self.img_encoder.load_state_dict(ckpt)
        self.img_head = nn.Sequential(
              nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=512, batch_first=True),
            	  nn.Linear(512, 3),)


        self.pose_img_fusion = PoseResGateFusion()




        self.action_decoder = DiffusionUNetPolicy(action_dim,horizon= num_action, n_obs_steps=num_obs, obs_feature_dim = obs_feature_dim, prediction = prediction)
        self.readout_embed = nn.Embedding(1, hidden_dim)

    def forward(self, img, goal, pose, actions = None, batch_size = 24):

        trans = pose[:,0,:3,2]
        rots = pose[:,0,:3,:2]
        rots = rots.reshape(rots.shape[0], rots.shape[1]*rots.shape[2])

        pose_feature = self.pose_encoder(trans, rots)

        # pose_feature = pose_feature.reshape(pose_feature.shape[0], pose_feature.shape[1]*pose_feature.shape[2])
        out = self.img_encoder(img, goal)
        out = self.img_head(out)
        img_feature = out.reshape(out.shape[0], out.shape[1]*out.shape[2])



        # feature = torch.cat([out, pose_feature], dim=1)
        feature = self.pose_img_fusion(pose_feature,img_feature )


        if actions is not None:
            loss = self.action_decoder.compute_loss(feature, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(feature)
            return action_pred