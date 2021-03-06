import numpy as np
import os
from keras.models import model_from_json
from pyrsgis import raster

def predict_image(original_image, model_type):
    """
    Channel order for numpy_image:
    - blue
    - green
    - red
    - red edge (B4)
    - red edge (B5)
    - red edge (B6)
    - near infra red (NIR)
    - vegetation red edge
    - shortwave infrared (SWIR) (B9)
    - shortwave infrared (SWIR) (B10)
    - CWC
    - LAI
    - MSI
    """
    if model_type == "Guinea":
        maxima = [1.08236387e+04, 1.11508086e+04, 1.13103369e+04, 5.60514697e+03,
                  6.80026270e+03, 7.73039795e+03, 1.39520293e+04, 7.77906787e+03,
                  1.61924541e+04, 1.57519648e+04, 1.38360262e-01, 5.02133465e+00,
                  1.07470360e+01]
        model = load_model(os.path.join(os.path.join("flask_server", "Models"), "model_guinea.json"),
                           os.path.join(os.path.join("flask_server", "Models"), "model_guinea.h5"))
    elif model_type == "Congo":
        maxima = [1.55391318e+04, 1.96239199e+04, 2.84899766e+04, 1.50214746e+04,
                  1.85321328e+04, 2.06404102e+04, 4.16918086e+04, 2.40694980e+04,
                  3.71772070e+04, 3.52320703e+04, 3.99795222e+00]
        model = load_model(os.path.join(os.path.join("flask_server", "Models"), "model_congo.json"),
                           os.path.join(os.path.join("flask_server", "Models"), "model_congo.h5"))

    else:
        raise Exception("Not a valid country")
    numpy_image = original_image.copy()
    numpy_image = np.moveaxis(numpy_image, 0, 2)

    for i in range(np.shape(numpy_image)[2]):
        numpy_image[..., i] = numpy_image[..., i] / maxima[i]

    width, height, _ = np.shape(numpy_image)
    prediction = np.zeros((width, height, 4))

    for i in range(0, width, 100):
        for j in range(0, height, 100):
            x_min = i
            x_max = i + 128
            y_min = j
            y_max = j + 128
            if x_max > width:
                x_min = width - 128
                x_max = width
            if y_max > height:
                y_min = height - 128
                y_max = height
            X_temp = numpy_image[x_min: x_max, y_min: y_max]
            pred = model.predict(np.array([X_temp]))
            prediction[x_min:x_max, y_min: y_max] += pred[0]

    final_prediction = np.argmax(prediction, axis=-1)

    del(model)
    return final_prediction

def load_model(path_to_json, path_to_weights):
    json_file = open(path_to_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(path_to_weights)
    return model

def load_image_from_path(all_bands_channels):
    dsi , data = raster.read(all_bands_channels, bands='all')
    data[data < 0] = 0
    return dsi, data

def load_image_from_paths(paths):
    all_bands_channels, msi, cwi, lai = paths[0], paths[1], paths[2], paths[3]
    ds , data = raster.read(all_bands_channels, bands='all')
    data = np.moveaxis(data, 0, 2)

    if cwi is not None:
        _, chan = raster.read(cwi, bands="all")
        cwi_data = np.reshape(chan, (np.shape(chan)[0], np.shape(chan)[1], 1))
        data = np.dstack([data, cwi_data])

    if lai is not None:
        _, chan = raster.read(lai, bands="all")
        lai_data = np.reshape(chan, (np.shape(chan)[0], np.shape(chan)[1], 1))
        data = np.dstack([data, lai_data])

    _, chan = raster.read(msi, bands='all')
    msi_data = np.reshape(chan, (np.shape(chan)[0], np.shape(chan)[1], 1))
    data = np.dstack([data, msi_data])

    return ds, data
