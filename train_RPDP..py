import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict


from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset.pose_data import RealWorldDataset_PoseRGBD, collate_fn2
from utils.training import set_seed, plot_history, sync_loss
from policy.policy import PoseRGBD_DP


from  policy.Utils import *

def process_rgbds(rgbs, depths,batch_size):
    As = []
    for i in range(len(rgbs)):
        rgb = rgbs[i]
        depth = depths[i]
        rgb_tensor = torch.as_tensor(rgb[:, :480, :], device='cuda', dtype=torch.float)
        depth = torch.as_tensor(depth[:, :480], device='cuda', dtype=torch.float)
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        # logging.info("depth processing done")
        K = np.array([[455.15264892578125, 0, 327.128662109375],
                      [0, 455.15264892578125, 240.3665771484375],
                      [0, 0, 1]])
        xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[
            0]
        xyz_map = xyz_map[:, :480, :]
        xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)

        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)
        xyz_map_tensor = xyz_map_tensor.permute(2, 0, 1).unsqueeze(0)

        rgb_tensor = F.interpolate(rgb_tensor, size=(160, 160), mode='bilinear', align_corners=False)
        xyz_map_tensor = F.interpolate(xyz_map_tensor, size=(160, 160), mode='bilinear', align_corners=False)

        A = torch.cat([rgb_tensor.cuda(), xyz_map_tensor.cuda()], dim=1).float()
        As.append(A)
    As = torch.cat(As, dim=0)
    return As


default_args = edict({
    "data_path": "data/push_block",
    "aug": False,
    "aug_jitter": False,
    "num_action": 20,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "ckpt_dir": "logs/collect_pens",
    "resume_ckpt": '/home/robotlab/sunhan/RISE/logs/push_block/policy_epoch_2_seed_233.ckpt',  ##NONE
    "resume_epoch": -1,
    "lr": 3e-4,
    "batch_size": 240,
    "num_epochs": 1000,
    "save_epochs": 100,
    "num_workers": 24,
    "seed": 233
})


def train(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset & dataloader
    print("Loading dataset ...")
    dataset = RealWorldDataset_PoseRGBD(
        path= args.data_path,
        split='train',
        num_obs=1,
        num_action=args.num_action,
        # aug=args.aug,
        normalize=True,
        gripper=False,
        sym_or=False,
    )

    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn2,
        sampler=sampler,
        shuffle=False)

    # policy
    print("Loading policy ...")
    policy = PoseRGBD_DP(
        num_action=20,
        input_dim=9,
        obs_feature_dim=128,
        action_dim=9,
        hidden_dim=512,
    ).to(device)
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # load checkpoint
    if args.resume_ckpt is not None:
        policy.load_state_dict(torch.load(args.resume_ckpt, map_location=device), strict=False)
        print("Checkpoint {} loaded.".format(args.resume_ckpt))

    # ckpt path
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    for param in policy.img_encoder.parameters():
      param.requires_grad = False

    # optimizer and lr scheduler
    print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=[0.95, 0.999], weight_decay=1e-6)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * args.num_epochs
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    # training
    train_history = []

    policy.train()
    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        print("Epoch {}".format(epoch))
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader)
        avg_loss = 0

        for data in pbar:
            obs_rgb = data['obs_rgb']
            obs_d = data['obs_d']
            goal_color = data['goal_color']
            goal_depth = data['goal_depth']
            obs_rgbd = process_rgbds(obs_rgb, obs_d, args.batch_size)
            goal_rgbd = process_rgbds(goal_color, goal_depth, args.batch_size)
            obs_source_pose = data['obs_source_pose']
            action_source_pose = data['action_source_pose']


            obs, action = obs_source_pose.to(device), action_source_pose.reshape(action_source_pose.shape[0],action_source_pose.shape[1],-1).to(device)

            # forward
            loss = policy(obs_rgbd,goal_rgbd, obs, action, batch_size=action.shape[0])
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()

        avg_loss = avg_loss / num_steps
        # sync_loss(avg_loss, device)
        train_history.append(avg_loss)

        print("Train loss: {:.6f}".format(avg_loss))
        if (epoch + 1) % args.save_epochs == 0:
            torch.save(
                policy.state_dict(),  ### sunhan
                os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, args.seed))
            )
            # plot_history(train_history, epoch, args.ckpt_dir, args.seed)

    torch.save(
        policy.state_dict(),
        os.path.join(args.ckpt_dir, "policy_last.ckpt")
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', type=str, help='data path', default='/home/sunh/1RobotMPL/HUAWEI/')
    parser.add_argument('--aug', action='store_true', help='whether to add 3D data augmentation')
    parser.add_argument('--aug_jitter', action='store_true', help='whether to add color jitter augmentation')
    parser.add_argument('--num_action', action='store', type=int, help='number of action steps', required=False,
                        default=20)
    parser.add_argument('--voxel_size', action='store', type=float, help='voxel size', required=False, default=0.005)
    parser.add_argument('--obs_feature_dim', action='store', type=int, help='observation feature dimension',
                        required=False, default=512)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden dimension', required=False, default=512)
    parser.add_argument('--nheads', action='store', type=int, help='number of heads', required=False, default=8)
    parser.add_argument('--num_encoder_layers', action='store', type=int, help='number of encoder layers',
                        required=False, default=4)
    parser.add_argument('--num_decoder_layers', action='store', type=int, help='number of decoder layers',
                        required=False, default=1)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='feedforward dimension', required=False,
                        default=2048)
    parser.add_argument('--dropout', action='store', type=float, help='dropout ratio', required=False, default=0.1)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='checkpoint directory', default='logs/mp')  #########################  3:soure pose -> gripper source gripper_w
    parser.add_argument('--resume_ckpt', action='store', type=str, help='resume checkpoint file', required=False,
                        default=None)
    parser.add_argument('--resume_epoch', action='store', type=int, help='resume from which epoch', required=False,
                        default=-1)
    parser.add_argument('--lr', action='store', type=float, help='learning rate', required=False, default=3e-4)

    parser.add_argument('--batch_size', action='store', type=int, help='batch size', required=False, default=160) # 80
    parser.add_argument('--num_epochs', action='store', type=int, help='training epochs', required=False, default=2000)
    parser.add_argument('--save_epochs', action='store', type=int, help='saving epochs', required=False, default=10)
    parser.add_argument('--num_workers', action='store', type=int, help='number of workers', required=False, default=0)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False, default=233)

    train(vars(parser.parse_args()))
