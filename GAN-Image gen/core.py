import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from generator import Generator
from discriminator import Discriminator

FOLDER = "images"
FOLDER_RESULT = "new_images"
MODELPTH = ''
BATCH_SIZE = 8
IMAGE_N = 10
IMAGE_SIZE = 64
LATENT_DIMENSION = 128
EPOCHS = 1000
LR = 0.0002

os.makedirs(FOLDER, exist_ok=True)
os.makedirs(FOLDER_RESULT, exist_ok=True)

def train(tensor_data):
    loader = DataLoader(tensor_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    Gen = Generator().to(torch.device('cpu'))
    Dis = Discriminator().to(torch.device('cpu'))
    crit = nn.BCELoss()
    optG = optim.Adam(Gen.parameters(), lr=LR, betas=(0.5, 0.999))
    optD = optim.Adam(Dis.parameters(), lr=LR, betas=(0.5, 0.999))

    real_label = 0.9
    fake_label = 0.1
    print(f"Adam Generator & Discriminator Training: {EPOCHS}\n")
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch + 1}')
        for real in loader:
            real = real.to(torch.device('cpu'))
            b = real.size(0)

            # Discriminator
            optD.zero_grad()
            d_real = Dis(real)
            loss_d_real = crit(d_real, torch.full((b,), real_label, device=torch.device('cpu')))
            
            z = torch.randn(b, LATENT_DIMENSION, device=torch.device('cpu'))
            fake = Gen(z)
            d_fake = Dis(fake.detach())
            loss_d_fake = crit(d_fake, torch.full((b,), fake_label, device=torch.device('cpu')))
            
            loss = loss_d_fake + loss_d_real
            loss.backward()
            optD.step()

            #Generator
            optG.zero_grad()
            g_loss = crit(Dis(fake), torch.full((b,), real_label, device=torch.device('cpu')))
            g_loss.backward()
            optG.step()
            print(f"Generator loss: {g_loss.item()}")
        if (epoch + 1) % 50 == 0:
            Gen.eval()
            with torch.no_grad():
                imgs = (((Gen(z).cpu() + 1) / 2).clamp(0,1) * 255).byte().permute(0, 2, 3, 1).numpy()
                for i, img in enumerate(imgs):
                    Image.fromarray(img).save(os.path.join(FOLDER_RESULT, f"gen_{epoch + 1}.png"))
            Gen.train()
    return Gen

if __name__ == "__main__":
    image_data = [os.path.join(FOLDER, file) for file in os.listdir(FOLDER) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    print('image path data imported')
    tensor_data = [torch.from_numpy((np.array(Image.open(image).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS), dtype=np.float32) / 127.5 - 1).transpose(2,0,1)) for image in image_data]
    print('tensor dataset ready > train session')
    if (MODELPTH == ""):
        Gen_model = train(tensor_data)
        torch.save(Gen_model.state_dict(), os.path.join(FOLDER_RESULT, "MODEL.pth"))
        MODELPTH = os.path.join(FOLDER_RESULT, "MODEL.pth")
        print('Model saved')
    Gen_model = Generator().to(torch.device('cpu'))
    Gen_model.load_state_dict(torch.load(MODELPTH, map_location=torch.device('cpu')))
    print('Model ready > try final result')
    Gen_model.eval()
    with torch.no_grad():
        for i in range(IMAGE_N):
            z = torch.randn(1, LATENT_DIMENSION, device=torch.device('cpu'))
            img = (((Gen_model(z).cpu().squeeze(0) + 1) / 2).clamp(0,1) * 255).byte().permute(1, 2, 0).numpy()
            Image.fromarray(img).save(os.path.join(FOLDER_RESULT, f"gen_final{i + 1}.png"))
    print('Finished')
