import os
import pydicom
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as tfs_v2

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Функция загрузки и нормализации DICOM-файлов
def load_dicom(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return Image.fromarray((img * 512).astype(np.uint8))

# Получение изображений из датасета
class SegmentDataset(data.Dataset):
    def __init__(self, path, transform_img=None, transform_mask=None):
        self.path = path
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        path = os.path.join(self.path, 'images')
        list_files = os.listdir(path)
        self.length = len(list_files)
        self.images = list(map(lambda _x: os.path.join(path, _x), list_files))

        path = os.path.join(self.path, 'masks')
        list_files = os.listdir(path)
        self.masks = list(map(lambda _x: os.path.join(path, _x), list_files))

    def __getitem__(self, item):
        path_img, path_mask = self.images[item], self.masks[item]

        if path_img.endswith('.dcm'):
            img = load_dicom(path_img).convert('L')
        else:
            img = Image.open(path_img).convert('L')

        mask = Image.open(path_mask).convert('L')  # grayscale

        if self.transform_img:
            img = self.transform_img(img)

        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask[mask < 250] = 0
            mask[mask >= 250] = 1

        return img, mask

    def __len__(self):
        return self.length


# Структура модели U-Net
class UNetModel(nn.Module):
    class _TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            return self.model(x)

    class _EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)
            return y, x

    class _DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)

        def forward(self, x, y):
            x = self.transpose(x)
            u = torch.cat([x, y], dim=1)
            u = self.block(u)
            return u

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_block1 = self._EncoderBlock(in_channels, 64)
        self.enc_block2 = self._EncoderBlock(64, 128)
        self.enc_block3 = self._EncoderBlock(128, 256)
        self.enc_block4 = self._EncoderBlock(256, 512)

        self.bottleneck = self._TwoConvLayers(512, 1024)

        self.dec_block1 = self._DecoderBlock(1024, 512)
        self.dec_block2 = self._DecoderBlock(512, 256)
        self.dec_block3 = self._DecoderBlock(256, 128)
        self.dec_block4 = self._DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)

        x = self.bottleneck(x)

        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)

        return self.out(x)


# Функция потерь (изменить на чистый Dice)
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = nn.functional.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2 * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score



# Функция обучения
def train_model(b_size=4, epochs=10):
    tr_img = tfs_v2.Compose([tfs_v2.ToTensor(), ])
    tr_mask = tfs_v2.Compose([tfs_v2.ToTensor(), ])

    d_train = SegmentDataset(r"dataset_seg", transform_img=tr_img, transform_mask=tr_mask)
    train_data = data.DataLoader(d_train, batch_size=b_size, shuffle=True)

    model = UNetModel(in_channels=1, num_classes=1)  # zero

    optimizer = optim.RMSprop(params=model.parameters(), lr=0.001)
    loss_1 = nn.BCEWithLogitsLoss()
    loss_2 = SoftDiceLoss()

    model.train()

    for _e in range(epochs):
        loss_mean = 0
        lm_count = 0

        train_tqdm = tqdm(train_data, leave=True)
        for x_train, y_train in train_tqdm:
            predict = model(x_train)
            loss = loss_1(predict, y_train) + loss_2(predict, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lm_count += 1
            loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
            train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

    # Сохранение весов и модели
    st = model.state_dict()
    torch.save(st, 'model_unet_seg.pth')


def predict_and_show(model, dicom_path):
    print('Предсказание моделью...')
    # Загрузка DICOM
    img_pil = load_dicom(dicom_path).convert('L')

    tr_img = tfs_v2.Compose([tfs_v2.ToTensor()])
    img_tensor = tr_img(img_pil).unsqueeze(0)

    # Предсказание
    model.eval()
    with torch.no_grad():
        predict = model(img_tensor)
        mask = torch.sigmoid(predict.squeeze(0).squeeze(0))

    # Визуализация
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img_pil), cmap='gray')
    plt.title('Исходное DICOM-изображение')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask.cpu().numpy(), cmap='gray')
    plt.title('Предсказанная маска')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return mask

if __name__ == "__main__":
    model = UNetModel(in_channels=1, num_classes=1)
    # Запуск обучения с параметрами
    # train_model(4, 10)

    # Предсказание моделью
    model.load_state_dict(torch.load('model_unet_seg.pth'))
    mask = predict_and_show(model, 'test.dcm')