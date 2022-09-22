import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np


class visualize_conv_layers:
    def __init__(self,model,img):
        self.image = image.img_to_array(image.load_img(img,target_size=(224,224)))
        self.models = model
    
    def visualization_model(self,):
        succesive_output = [layer.output for layer in self.models.layers]
        vis_model = tf.keras.models.Model(inputs=self.models.input,outputs=succesive_output)
        inp_img = np.expand_dims(self.image,axis=0)
        self.succesive_feature_maps = vis_model.predict(inp_img)
        self.visualize(self.succesive_feature_maps)

    def visualize(self,succesive_feature_maps):
        layer_names = [layer.name for layer in self.models.layers]
        # Display the representations
        for layer_name, feature_map in zip(layer_names, self.succesive_feature_maps):
            if len(feature_map.shape) == 4:

                # Just do this for the conv / maxpool layers, not the fully-connected layers
                n_features = feature_map.shape[-1]  # number of features in feature map

                # The feature map has shape (1, size, size, n_features)
                size = feature_map.shape[1]
                
                # Tile the images in this matrix
                display_grid = np.zeros((size, size * n_features))
                for i in range(n_features):
                    x = feature_map[0, :, :, i]
                    x -= x.mean()
                    x /= x.std()
                    x *= 64
                    x += 128
                    x = np.clip(x, 0, 255).astype('uint8')
                    
                    # Tile each filter into this big horizontal grid
                    display_grid[:, i * size : (i + 1) * size] = x
                    
                # Display the grid
                scale = 20. / n_features
                plt.figure(figsize=(scale * n_features, scale))
                plt.title(layer_name)
                plt.grid = False
                plt.imshow(display_grid, aspect='auto', cmap='viridis')