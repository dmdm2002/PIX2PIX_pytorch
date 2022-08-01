import Options
import os
import re
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from Modeling.Generator import Gen
from Modeling.Discrminator import disc
from utils.DataLoader import Loader
from utils.func import ImagePool
from utils.Displayer import LossDisplayer
from torch.autograd import Variable

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class train(Options.param):
    def __init__(self):
        super(train, self).__init__()
        # self.writer = SummaryWriter()

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    def sampling(self, output, transform_to_image, name, epoch, type):
        os.makedirs(f"{self.OUTPUT_SAMPLE}/sample_{type}/{epoch}", exist_ok=True)
        output = transform_to_image(output.squeeze())
        output.save(f'{self.OUTPUT_SAMPLE}/sample_{type}/{epoch}/{name}_{type}.png')

    def run(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print(f'[device] : {device}')
        print('--------------------------------------------------------------------------------')
        # writer = SummaryWriter()

        # 1. Model Build
        num_blocks = 6 if self.SIZE <= 256 else 8

        generator = Gen().to(device)
        discriminator = disc().to(device)

        os.makedirs(f"{self.OUTPUT_CKP}/ckp", exist_ok=True)
        os.makedirs(f"{self.OUTPUT_SAMPLE}/sample_A2B", exist_ok=True)
        os.makedirs(f"{self.OUTPUT_SAMPLE}/sample_A2B2A", exist_ok=True)

        # 2. Load CKP
        if self.CKP_LOAD:
            ckp = torch.load(self.OUTPUT_CKP, map_location=device)
            generator.load_state_dict(ckp["generator_state_dict"])
            discriminator.load_state_dict(ckp["discriminator_state_dict"])
            epoch = ckp["epoch"]
        else:
            generator.apply(self.weights_init_normal)
            discriminator.apply(self.weights_init_normal)

        generator.train()
        discriminator.train()

        # 3. DataLoader
        transform = transforms.Compose(
            [
                transforms.Resize((self.SIZE, self.SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        transforms_to_image = transforms.Compose(
            [
                transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
                transforms.ToPILImage(),
            ]
        )

        # pool_fake_A = ImagePool(self.POOL_SIZE)
        # pool_fake_B = ImagePool(self.POOL_SIZE)

        # 4. LOSS
        criterion_GAN = torch.nn.BCELoss()
        criterion_pixelwise = torch.nn.L1Loss()

        disp = LossDisplayer(["loss_G", "loss_D"])
        summary = SummaryWriter()

        # 5. Optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=self.LR, betas=(self.B1, self.B2))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=self.LR, betas=(self.B1, self.B2))

        # patch 수
        patch = (1, self.SIZE // 2 ** 4, self.SIZE // 2 ** 4)

        for epoch in range(self.EPOCH):
            print(f"|| Now Epoch : [{epoch}/{self.EPOCH}] ||")

            dataset = Loader(self.DATASET_PATH, self.DATA_STYPE, transform)
            dataloader = DataLoader(dataset=dataset, batch_size=self.BATCHSZ, shuffle=False)

            for idx, (real_A, real_B, name) in enumerate(dataloader):
                ba_si = real_A.size(0)
                real_A = real_A.to(device)
                real_B = real_B.to(device)

                real_label = torch.ones(ba_si, *patch, requires_grad=False).to(device)
                fake_label = torch.zeros(ba_si, *patch, requires_grad=False).to(device)

                # Train Generator
                optimizer_G.zero_grad()

                fake_B = generator(real_A)
                pred_fake = discriminator(fake_B, real_B)
                loss_GAN = criterion_GAN(pred_fake, real_label)

                # Pixel-wise Loss
                loss_pixel = criterion_pixelwise(fake_B, real_B)

                # Total Loss
                loss_G = loss_GAN + self.LAMDA_PIXEL * loss_pixel

                loss_G.backward()

                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()

                # Real Loss
                pred_real = discriminator(real_B, real_A)
                loss_real = criterion_GAN(pred_real, real_label)

                pred_fake = discriminator(fake_B.detach(), real_A)
                loss_fake = criterion_GAN(pred_fake, fake_label)

                loss_D = (loss_real + loss_fake) * 0.5

                loss_D.backward()
                optimizer_D.step()

                print(f'===> Epoch[{epoch}/{self.EPOCH}] ({idx}/{len(dataloader)}): Loss_G : {loss_G} | Loss_D : {loss_D}')

                disp.record([loss_G, loss_D])

                if idx % 100 == 0:
                    name = name[0].split("\\")[-1]
                    name = re.compile(".png").sub('', name)

                    self.sampling(fake_B[0], transforms_to_image, name, epoch, "A2B")

            avg_losses = disp.get_avg_losses()
            summary.add_scalar("Loss_G", avg_losses[0], epoch)
            summary.add_scalar("Loss_D", avg_losses[1], epoch)

            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(f"{self.OUTPUT_CKP}/ckp", f"{epoch}.pth"),
            )
