from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import model_from_yaml
import numpy as np
import sys

#print(sys.argv)
img_path = sys.argv[1]

try:
    # load YAML and create model
    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
except:
    model = ResNet50(weights='imagenet')
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
