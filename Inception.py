import tensorflow as tf
from download import maybe_download_and_extract
import os
import pickle
import cv2
import matplotlib.pyplot as plt

# Internet URL for the tar-file with the Inception model.
# Note that this might change in the future and will need to be updated.
data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

# Directory to store the downloaded data.
data_dir = "inception/"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "classify_image_graph_def.pb"

CACHE_FILE_NAME = './inception/cache.pkg'


def maybe_download():
    """
    Download the Inception model from the internet if it does not already
    exist in the data_dir. The file is about 85 MB.
    """

    print("Downloading Inception v3 Model ...")
    maybe_download_and_extract(url=data_url, download_dir=data_dir)


class Inception:
    # Name of the tensor for feeding the input image as jpeg.
    tensor_name_input_jpeg = "DecodeJpeg/contents:0"

    tensor_name_transfer_layer = "mixed_10/join:0"

    tensor_name_input_image = "DecodeJpeg:0"

    def __init__(self):
        # Create a new TensorFlow computational graph.
        self.graph = tf.Graph()

        # Set the new graph as the default.
        with self.graph.as_default():
            # Open the graph-def file for binary reading.
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # The graph-def is a saved copy of a TensorFlow graph.
                # First we need to create an empty graph-def.
                graph_def = tf.GraphDef()

                # Then we load the proto-buf file into the graph-def.
                graph_def.ParseFromString(file.read())

                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')

        # Get the tensor for the last layer of the graph, aka. the transfer-layer.
        self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)

        # Get the number of elements in the transfer-layer.
        self.transfer_len = self.transfer_layer.get_shape()[3]

        # Create a TensorFlow session for executing the graph.
        self.session = tf.Session(graph=self.graph)

        # Create a TensorFlow session for executing the graph.
        self.session = tf.Session(graph=self.graph)

    def close(self):
        """
        Call this function when you are done using the Inception model.
        It closes the TensorFlow session to release its resources.
        """

        self.session.close()

    def _create_feed_dict(self, image_path=None, image=None):
        """
        Create and return a feed-dict with an image.

        :param image_path:
            The input image is a jpeg-file with this file-path.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            Dict for feeding to the Inception graph in TensorFlow.
        """

        # if image is not None:
        # Image is passed in as a 3-dim array that is already decoded.
        # feed_dict = {self.tensor_name_input_image: image}
        if image is not None:
            # Image is passed in as a 3-dim array that is already decoded.
            feed_dict = {self.tensor_name_input_image: image}
        elif image_path is not None:
            # Read the jpeg-image as an array of bytes.
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            # Image is passed in as a jpeg-encoded image.
            feed_dict = {self.tensor_name_input_jpeg: image_data}

        else:

            raise ValueError("Either image or image_path must be set.")

        return feed_dict

    def extract_feature(self, image_path=None, image=None):
        feed_dict = self._create_feed_dict(image_path=image_path)
        pred = self.session.run(self.tensor_name_transfer_layer, feed_dict=feed_dict)
        return pred

    def build_cache(self, image_dir):
        cache = {}
        img_files = os.listdir(image_dir)
        for img in img_files:
            if img not in cache:
                if '.csv' in img:
                    continue

                print(os.path.join(image_dir, img))
                features = self.extract_feature(image_path=os.path.join(image_dir, img))
                cache[img] = features

        with open(CACHE_FILE_NAME, 'wb') as cache_file:
            pickle.dump(cache, cache_file)

        return cache

    def read_cache(self):
        if not os.path.exists(CACHE_FILE_NAME):
            raise RuntimeError('file not found')

        with open(CACHE_FILE_NAME, 'rb') as cache_file:
            return pickle.load(cache_file)

    def extract_features(self, image_dir):
        if os.path.exists(CACHE_FILE_NAME):
            print('Reading from cache.....')
            return self.read_cache()

        print('Generate and cache features ....')
        self.build_cache(image_dir)

    def build_input_cache(self, image_dir):
        cache = {}
        names = os.listdir(image_dir)
        for name in names:
            if 'csv' in name:
                continue

            if name not in cache:
                full_path = os.path.join(image_dir, name)
                print(full_path)
                img = plt.imread(full_path)
                cache[name] = img

        with open(CACHE_FILE_NAME, 'wb') as cache_file:
            pickle.dump(cache, cache_file)


########################################################################
# Example usage.

if __name__ == '__main__':
    print(tf.__version__)

    # Download Inception model if not already done.
    maybe_download()

    # Load the Inception model so it is ready for classifying images.
    model = Inception()

    # Path for a jpeg-image that is included in the downloaded data.
    image_path = os.path.join(data_dir, '/home/upul/datasets/udacity/debug/1479498372942264998.jpg')

    # Use the Inception model to classify the image.
    # pred = model.extract_feature(image_path=image_path)

    # model.build_cache('/home/upul/datasets/udacity/debug/')

    # print(model.read_cache()['1479498372942264998.jpg'].shape)

    model.extract_features('/home/upul/datasets/udacity/debug/1')

    # Print the scores and names for the top-10 predictions.
    # model.print_scores(pred=pred, k=10)

    #img = plt.imread('/home/upul/datasets/udacity/debug/1479498372942264998.jpg')
    #x = model.extract_feature(image=img)
    #print(x.shape)
    #model.build_input_cache('/home/upul/datasets/udacity/debug/1')

    # Close the TensorFlow session.
    #model.close()

    # Transfer Learning is demonstrated in Tutorial #08.

########################################################################
