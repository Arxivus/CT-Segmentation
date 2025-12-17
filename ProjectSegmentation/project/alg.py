from PIL import Image
import os
import pandas as pd
from medpy.metric import dc, jc
from medpy.io import load
from sklearn.model_selection import train_test_split as tts
from tensorflow.python.keras.models import load_model
from tqdm import tqdm
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2


SEED = 42

# Подготовка данных ==========================================================

def path(fname, num):
    return f'../Archive/CHAOS_Train_Sets/Train_Sets/CT/' + fname

def get_data(num):
    X_path = path('', num)
    X_filenames = []
    y_filenames = []
    for directory in tqdm(os.listdir(X_path)[:-1], position=0):
        for dirName, subdirList, fileList in os.walk(X_path+directory):
            for filename in fileList:
                if ".dcm" in filename.lower():
                    X_filenames.append(os.path.join(dirName, filename))
                if ".png" in filename.lower():
                    y_filenames.append(os.path.join(dirName, filename))
    assert len(X_filenames) == len(y_filenames)
    return sorted(X_filenames), sorted(y_filenames)

X_filenames1, y_filenames1 = get_data(1)

X_filenames = X_filenames1
y_filenames = y_filenames1

len(X_filenames), len(y_filenames)

# Буферизация изображений =====================================================================

def buffer_imgs(filenames, is_dicom, folder='buffer'):
    files = []
    if not os.path.exists(folder):
        os.makedirs(folder)
    for filename in tqdm(filenames, position=0):
        img, header = load(filename)
        pil = Image.fromarray(img.squeeze())
        # fname = folder + '/' + filename.replace('/', '-')
        fname = folder + '/' + filename.replace('\\', '-').replace('/', '-').replace(':', '-').lstrip('-')
        if is_dicom:
            fname = fname+'.tiff'
            pil.save(fname, 'TIFF', compression='none')
        else:
            pil.save(fname, fname.split('.')[-1], compression='none')
        files.append(fname)
    return pd.DataFrame(files)

# Разделение данных на тренировочные и тестовые ====================================================

X = buffer_imgs(X_filenames, True)
y = buffer_imgs(y_filenames, False)

X.shape, y.shape

def show_img(n):
    plt.figure(figsize=(5,5))
    plt.imshow(Image.open(X[0][n]))
    plt.imshow(Image.open(y[0][n]), alpha=0.4)
    plt.title(n)

for i in np.random.choice(np.arange(y.shape[0]), 10):
    show_img(i)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, shuffle=True, random_state=SEED)

# Метрики модели UNet =============================================================

def dice_coef(y_true, y_pred):
    smooth = 1e-20
    y_true_f = K.cast(y_true, 'float32')
    intersection = K.sum(y_true_f * y_pred)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def jaccard_coef(y_true, y_pred):
    smooth = 1e-20
    y_true_f = K.cast(y_true, 'float32')
    intersection = K.sum(y_true_f * y_pred)
    union = K.sum(y_true_f + y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def jaccard_coef_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)

# Подготовка генераторов данных ================================================

X_train.shape, y_train.shape
w_size = np.array(Image.open(X[0][0])).shape[0]

model_1 = load_model('fix_unet.h5', custom_objects={
    'dice_coef': dice_coef,
    'dice_coef_loss': dice_coef_loss,
    'jaccard_coef': jaccard_coef,
    'jaccard_coef_loss': jaccard_coef_loss
})

X_tr, X_val, y_tr, y_val = tts(X_train, y_train, test_size=0.1, shuffle=True, random_state=SEED)

gen_train_params = {
    'rotation_range':10,
    'fill_mode':'reflect',
}

idg_train_data = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
                                    **gen_train_params,
                             )
idg_train_mask = ImageDataGenerator(rescale=1./255,
                                    **gen_train_params)

train_gen_params = {
    'x_col': 0,
    'target_size': (512, 512),
    'color_mode': 'grayscale',
    'batch_size': 4,
    'class_mode': None,
    'shuffle': True,
    'seed': SEED,
}
val_gen_params = train_gen_params.copy()
val_gen_params['shuffle'] = False
val_gen_params['batch_size'] = 1

data_train_generator = idg_train_data.flow_from_dataframe(X_tr, **train_gen_params)
mask_train_generator = idg_train_mask.flow_from_dataframe(y_tr, **train_gen_params)
train_generator = zip(data_train_generator, (x.astype(np.uint8) for x in mask_train_generator))

idg_test_data = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True
                                  )
idg_test_mask = ImageDataGenerator(rescale=1./255)
data_test_generator = idg_test_data.flow_from_dataframe(X_val,**val_gen_params)
mask_test_generator = idg_test_mask.flow_from_dataframe(y_val, **val_gen_params)
test_generator = zip(data_test_generator, (x.astype(np.uint8) for x in mask_test_generator))

# Предсказания модели (поиск контуров) ============================================================

def evaluate(x_names, y_names, set_name='evaluating', plot_pairs=0):
    val_gen_params['batch_size'] = 1
    print(set_name)
    dices = []
    jccrs = []

    data_g = idg_test_data.flow_from_dataframe(x_names, **val_gen_params)
    mask_g = idg_test_mask.flow_from_dataframe(y_names, **val_gen_params)

    for i, (image, mask) in enumerate(zip(tqdm(data_g), mask_g)):
        if i >= len(x_names):
            break
        if mask.max() == 0:
            continue

        p = model_1.predict(image).astype('uint8')
        dice = dc(p, mask)
        dices.append(dice)

        try:
            jccr = jc(p, mask)
        except ZeroDivisionError:
            jccr = 1
        jccrs.append(jccr)

        if plot_pairs > 0:
            plot_pairs -= 1

            mask_8uc1 = (mask.squeeze() * 255).astype(np.uint8)
            pred_8uc1 = (p.squeeze() * 255).astype(np.uint8)

            contours_true, _ = cv2.findContours(mask_8uc1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_pred, _ = cv2.findContours(pred_8uc1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(image.squeeze(), cmap='gray')
            ax1.imshow(mask.squeeze(), alpha=0.5, cmap='autumn')
            for contour in contours_true:
                ax1.plot(contour[:, 0, 0], contour[:, 0, 1], 'b', linewidth=2)
            ax1.set_title('Истинная маска')

            ax2.imshow(image.squeeze(), cmap='gray')
            ax2.imshow(p.squeeze(), alpha=0.5, cmap='autumn')
            for contour in contours_pred:
                ax2.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)
            ax2.set_title(f'Предсказанная, Dice={dice:.2f}')
            plt.show()

    mean_dice = np.mean(dices)
    mean_jccr = np.mean(jccrs)

    print('Средний Dice:', mean_dice)
    print('Средний Jaccard:', mean_jccr)
    print('-------------')

    return mean_dice, mean_jccr

evaluate(X_tr, y_tr, 'TR SET', 10)


