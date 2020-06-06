import numpy as np
from PIL import Image, ImageFont, ImageDraw
import datetime
import os
import io
import matplotlib.pyplot as plt


######################### Fonctions auxiliaires #########################

def get_bar_plot(output, hex0, hex1, hex2):
    nb_total = np.shape(output)[0] * np.shape(output)[1]
    nb_vir = np.sum(output == 1) * 100 / nb_total
    nb_def = np.sum(output == 2) *100/ nb_total
    nb_other = (np.sum(output == 3) + np.sum(output == 0))*100 / nb_total
    y = [nb_vir, nb_def, nb_other]
    y_text = ["%.2f" % nb_vir, "%.2f" % nb_def, "%.2f" % nb_other]

    plt.figure(num=None, figsize=(4, 8), dpi=300)
    ax = plt.subplot(111)
    ax.bar(x=[0,1,2], height=y, color=[hex0, hex1, hex2])
    plt.xticks([0,1,2], ["Intact\n Forest", "Degraded\n Forest", "Other"])
    ax.tick_params(axis='x',  labelsize=15)
    plt.ylim(0,np.max(y) + 0.1)
    plt.yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i in range(0,3):
        ax.text(x=i-0.2, y =y[i]+0.01, s=f"{y_text[i]}" , fontsize=15)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf

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

    image = Image.fromarray(new_image.astype('uint8'), 'RGB')
    return image

def convert_image_msi(img):
    row = np.expand_dims(normalize_img(img[-1,:,:]), axis = 2)
    image = Image.fromarray(np.uint8(255*np.concatenate((row, row, row), axis=2)), 'RGB')
    return image

def convert_image_rgb(img):
    red_row = np.expand_dims(normalize_img(img[2,:,:]), axis = 2)
    green_row = np.expand_dims(normalize_img(img[1,:,:]), axis = 2)
    blue_row = np.expand_dims(normalize_img(img[0,:,:]), axis = 2)
    image = Image.fromarray(np.uint8(255*np.concatenate((red_row, green_row, blue_row), axis=2)), 'RGB')
    return image

def convert_image_infra(img):
    red_row = np.expand_dims(normalize_img(img[6,:,:]), axis = 2)
    green_row = np.expand_dims(normalize_img(img[8,:,:]), axis = 2)
    blue_row = np.expand_dims(normalize_img(img[2,:,:]), axis = 2)
    image = Image.fromarray(np.uint8(255*np.concatenate((red_row, green_row, blue_row), axis=2)), 'RGB')
    return image

def mix_images(im1, img2):
    im2 = img2.resize(im1.size)
    mask = Image.new("L", im1.size, 128)
    composite = Image.composite(im1, im2, mask)
    return composite

def add_legend(img):
    legend = create_legend_bar(img.width)
    image = get_concat_v_cut(img, legend)
    return image

def get_kpis(output):
    nb_vir = np.sum(output == 1)
    nb_def = np.sum(output == 2)
    nb_other = np.sum(output == 3) + np.sum(output == 0)
    nb_total = nb_vir + nb_def + nb_other
    nb_vir_string = "%.2f" % (nb_vir/nb_total*100)
    nb_def_string = "%.2f" % (nb_def/nb_total*100)
    nb_other_string = "%.2f" %(nb_other/nb_total*100)
    result = "Date of submission: " + str(datetime.datetime.now().date()) + \
             " ; Total area: "+ str(int(nb_total/10000)) + "km² ; " \
              "Intact forest: " + nb_vir_string + \
              "% ; Degraded forest: " + nb_def_string + "% ; " \
              "Other: "  + nb_other_string + "%"
    return result

def generate(input, output, hex0, hex1, hex2):
    kpis = get_kpis(output)
    bar_plot = get_bar_plot(output, hex0, hex1, hex2)
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
    all_images = [mask, msi, rgb, infra, mask_msi, mask_rgb, msi_rgb, mask_infra, rgb_infra, msi_infra, mask_msi_infra, mask_rgb_infra, msi_rgb_infra, msi_rgb_mask, all]
    images_with_legend = []
    for im in all_images:
        images_with_legend.append(add_legend(im))
    images_with_legend.append(kpis)
    images_with_legend.append(bar_plot)
    return images_with_legend


def create_legend_bar(width):
    step = 4

    height = int(width * 0.1)
    half_height = int(height / 4)
    interval = int(height / 50)
    width_step = int((width * 0.2) / step)
    label = np.zeros((height, width)) + 255
    #TODO Assure that it works also with small images
    for j in range(int(width * 0.2)):
        for i in range(half_height - interval, half_height + interval):
            label[i][j + 50] = 0
        if j % width_step < int(interval / 2) or j % width_step > width_step - int(interval / 2):
            for i in range(half_height - 5 * interval, half_height + 5 * interval):
                label[i][j + 50] = 0
    path_to_font = os.path.join(os.path.join("flask_server","static"), "fonts")
    font = ImageFont.truetype(os.path.join(path_to_font, "arial.ttf"), int(width_step / 5))
    img = Image.fromarray(label.astype('uint8'), 'L')
    draw = ImageDraw.Draw(img)
    draw.text((25 + (0 * width_step), half_height + 6 * interval), str(0), font=font)
    #TODO Assure that it works also with small images

    for j in range(1, step):
        draw.text((25 + (j * width_step) - interval, half_height + 6 * interval), str(width_step * j), font=font)

    draw.text((25 + (step * width_step) - interval, half_height + 6 * interval), str(width_step * step) + " km", font=font)

    return img

def get_concat_v_cut(im1, im2):
    dst = Image.new('RGB', (min(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
