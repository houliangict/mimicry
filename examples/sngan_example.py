"""
Typical usage example.
"""
import sys
sys.path.append('..')

import torch
import torch.optim as optim
import torch_mimicry as mmc
from torch_mimicry.nets import sngan

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--log_dir', type=str, default='./logs/gan')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.0)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--n_steps', type=int, default=100000)
parser.add_argument('--n_dis', type=int, default=5)
args = parser.parse_args()

if __name__ == "__main__":
    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root='./datasets', name=args.dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=4)

    # Define models and optimizers
    netG = sngan.SNGANGenerator32().to(device)
    netD = sngan.SNGANDiscriminator32().to(device)
    optD = optim.Adam(netD.parameters(), args.lr, betas=(args.beta1, args.beta2))
    optG = optim.Adam(netG.parameters(), args.lr, betas=(args.beta1, args.beta2))

    # Start training
    trainer = mmc.training.Trainer(netD=netD,
                                   netG=netG,
                                   optD=optD,
                                   optG=optG,
                                   n_dis=args.n_dis,
                                   num_steps=args.n_steps,
                                   dataloader=dataloader,
                                   log_dir=args.log_dir,
                                   device=device)
    trainer.train()

    # Evaluate fid
    mmc.metrics.evaluate(metric='fid',
                         log_dir=args.log_dir,
                         netG=netG,
                         dataset=args.dataset,
                         num_real_samples=10000,
                         num_fake_samples=10000,
                         evaluate_step=args.n_steps,
                         device=device,
                         split='test')

    # Evaluate kid
    mmc.metrics.evaluate(metric='kid',
                         log_dir=args.log_dir,
                         netG=netG,
                         dataset=args.dataset,
                         num_samples=10000,
                         evaluate_step=args.n_steps,
                         device=device,
                         split='test')

    # Evaluate inception score
    mmc.metrics.evaluate(metric='inception_score',
                         log_dir=args.log_dir,
                         netG=netG,
                         num_samples=50000,
                         evaluate_step=args.n_steps,
                         device=device)
