import Options
import os
import re
import glob
import torch
import torchvision.transforms as transforms
import random
import PIL.Image as Image

from Modeling.Generator import Gen


class test(Options.param):
    def __init__(self):
        super(test, self).__init__()

    def shuffle_folder(self, A_folders, B_folders):
        random.shuffle(B_folders)
        for i in range(len(A_folders)):
            if A_folders[i] == B_folders[i]:
                return self.shuffle_folder(A_folders, B_folders)
        return B_folders

    def Shuffle_inner_class(self, A_folders, B_folders):
        A_result = []
        B_result = []

        for imgs in A_folders:
            A_imgs = glob.glob(f'{imgs}/*')
            B_imgs = glob.glob(f'{imgs}/*')
            B_imgs = self.shuffle_folder(A_imgs, B_imgs)

            A_result = A_result + A_imgs
            B_result = B_result + B_imgs

        return A_result, B_result

    def np2img(self, output, transform_to_image, name, epoch, type):
        os.makedirs(f"{self.OUTPUT_SAMPLE}/{epoch}/A/{type}", exist_ok=True)
        output = transform_to_image(output.squeeze())
        output.save(f'{self.OUTPUT_SAMPLE}/{epoch}/A/{type}/{name}.png')

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'[device] : {device}')
        print('--------------------------------------------------------------------------------')

        # 1. Model Build
        num_blocks = 6 if self.SIZE <= 256 else 8

        netG_A2B = Gen(num_blocks).to(device)
        netG_B2A = Gen(num_blocks).to(device)

        # 2. Load CKP
        checkpoint = torch.load(f'{self.OUTPUT_CKP}/ckp/199.pth', map_location=device)
        netG_A2B.load_state_dict(checkpoint["netG_A2B_state_dict"])
        netG_B2A.load_state_dict(checkpoint["netG_B2A_state_dict"])

        netG_A2B.eval()
        netG_B2A.eval()

        # 3. Load DataSets
        transform_to_tensor = transforms.Compose(
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

        # 4. Setting Folder
        os.makedirs(f'{self.OUTPUT_TEST}/A2B', exist_ok=True)
        os.makedirs(f'{self.OUTPUT_TEST}/B2A', exist_ok=True)

        os.makedirs(f'{self.OUTPUT_TEST}/A2B_LOSS', exist_ok=True)
        os.makedirs(f'{self.OUTPUT_TEST}/B2A_LOSS', exist_ok=True)

        test_list = [["A2B", netG_A2B], ["B2A", netG_B2A]]

        # LOOP
        for folder_name, model in test_list:
            print(f'[Folder Name] : {folder_name}')

            A_folders = glob.glob(f'{os.path.join(self.DATASET_PATH, self.DATA_STYPE[0])}/*')
            B_folders = glob.glob(f'{os.path.join(self.DATASET_PATH, self.DATA_STYPE[0])}/*')

            A_result, B_result = self.Shuffle_inner_class(A_folders, B_folders)

            if folder_name == "A2B":
                image_path_list = A_result
            else:
                image_path_list = B_result

            for idx, img_path in enumerate(image_path_list):
                print(f'|| {idx}/{len(image_path_list)} ||')
                image = Image.open(img_path)
                image = transform_to_tensor(image).unsqueeze(0)
                image = image.to(device)

                output = model(image)

                name = img_path.split("\\")[-1]
                name = re.compile(".png").sub('', name)

                self.np2img(output, transforms_to_image, name, '199', folder_name)



if __name__ == "__main__":
    a = test()
    a.run()
