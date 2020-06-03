import numpy as np
from PIL import Image
import datetime

######################### Fonctions auxiliaires #########################

def normalize_img(img):
  scale = 1/(np.max(img) - np.min(img))
  return scale * (img - np.max(img))

def hex_to_array(hex):
    if hex[0] == "#":
        return [int(hex[1:][i:i+2], 16) for i in (0, 2, 4)]
    else:
        return [int(hex[i:i+2], 16) for i in (0, 2, 4)]

def convert_mask_to_image(img, hex0, hex1, hex2):
    new_image = np.zeros((img.shape[0], img.shape[1], 3))
    new_image[img == 1] =  np.uint8(hex_to_array(hex0))
    new_image[img == 2] =  np.uint8(hex_to_array(hex1))
    new_image[img == 3] =  np.uint8(hex_to_array(hex2))

    return Image.fromarray(new_image.astype('uint8'), 'RGB')

def convert_image_msi(img):
    row = np.expand_dims(normalize_img(img[10,:,:]), axis = 2)
    return Image.fromarray(np.uint8(255*np.concatenate((row, row, row), axis=2)), 'RGB')

def convert_image_rgb(img):
    f_row = np.expand_dims(normalize_img(img[0,:,:]), axis = 2)
    s_row = np.expand_dims(normalize_img(img[1,:,:]), axis = 2)
    t_row = np.expand_dims(normalize_img(img[2,:,:]), axis = 2)
    return Image.fromarray(np.uint8(255*np.concatenate((f_row, s_row, t_row), axis=2)), 'RGB')

def convert_image_infra(img):
    f_row = np.expand_dims(normalize_img(img[6,:,:]), axis = 2)
    s_row = np.expand_dims(normalize_img(img[8,:,:]), axis = 2)
    t_row = np.expand_dims(normalize_img(img[2,:,:]), axis = 2)
    return Image.fromarray(np.uint8(255*np.concatenate((f_row, s_row, t_row), axis=2)), 'RGB')

def mix_images(im1, img2):
    im2 = img2.resize(im1.size)
    mask = Image.new("L", im1.size, 128)
    return Image.composite(im1, im2, mask)

def get_kpis(output):
    nb_vir = np.sum(output == 1)
    nb_def = np.sum(output == 2)
    nb_other = np.sum(output == 3) + np.sum(output == 0)
    nb_total = nb_vir + nb_def + nb_other
    return "Date of submission: " + str(datetime.datetime.now().date()) + " ; Total area: "+ str(nb_total*100) + "m² ; Intact forest: " + str(nb_vir/nb_total*100)[:4] + "% ; Degraded forest: "+ str(nb_def/nb_total*100)[:4] + "% ; Other: " + str(nb_other/nb_total*100)[:4]+"%"

def generate(input, output, hex0, hex1, hex2):
    kpis = get_kpis(output)
    mask = convert_mask_to_image(output, str(hex0), str(hex1), str(hex2))
    msi = convert_image_msi(input)
    rgb = convert_image_rgb(input)
    infra = convert_image_infra(input)
    mask_msi = mix_images(mask, msi)
    mask_rgb = mix_images(rgb, mask)
    msi_rgb = mix_images(msi, rgb)
    mask_infra = mix_images(mask, infra)
    rgb_infra = mix_images(rgb, infra)
    msi_infra = mix_images(msi, infra)
    mask_msi_infra = mix_images(mask_infra, msi)
    mask_rgb_infra = mix_images(rgb_infra, mask)
    msi_rgb_infra = mix_images(msi_infra, rgb)
    msi_rgb_mask = mix_images(mask_rgb, msi)
    all = mix_images(msi_rgb_mask, infra)
    return mask, msi, rgb, infra, mask_msi, mask_rgb, msi_rgb, mask_infra, rgb_infra, msi_infra, mask_msi_infra, mask_rgb_infra, msi_rgb_infra, msi_rgb_mask, all, kpis
