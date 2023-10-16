"""
Mask R-CNN
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 rebar.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 rebar.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 rebar.py train --dataset=/path/to/balloon/dataset --weights=imagenet
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")

# Root directory of the project
# ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# Go up two levels to the repo root
ROOT_DIR = os.path.join(ROOT_DIR, "..", "..")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/rebar/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = [
    "newset4-2_00000",
    "newset4-2_00001",
    "newset4-2_00002",
    "newset4-2_00003",
    "newset4-2_00004",
    "newset4-2_00005",
    "newset4-2_00006",
    "newset4-2_00007",
    "newset4-2_00008",
    "newset4-2_00009",
    "newset4-2_00010",
    "newset1_00001",
    "newset1_00002",
    "newset1_00003",
    "newset1_00004",
    "newset1_00005",
    "newset1_00006",
    "newset1_00007",
    "newset1_00008",
    "newset1_00009",
    "newset1_00010",
    "bar_00001",
    "bar_00002",
    "bar_00003",
    "bar_00004",
    "bar_00005",
    "bar_00006",
    "bar_00007",
    "bar_00008",
    "bar_00009",
    "bar_00010",
]

TRAIN_PORT = 0.99

############################################################
#  Configurations
############################################################

class RebarConfig(Config):
    """Configuration for training on the rebar segmentation dataset."""
    # Give the configuration a recognizable name
    # Give the configuration a recognizable name
    NAME = "rebar"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0
    DETECTION_MAX_INSTANCES = 50


class RebarInferenceConfig(RebarConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    # IMAGE_RESIZE_MODE = "pad64"
    # Skip detections with < 90% confidence
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.9

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3


############################################################
#  Dataset
############################################################

class RebarDataset(utils.Dataset):

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        # conver to gray
        image = np.uint8(skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255)
        return image

    def load_rebar(self, dataset_dir, subset):
        """Load a subset of the rebar dataset.

        dataset_dir: Root directory of the dataset
        subset: * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset rebar, and the class rebar
        self.add_class("rebar", 1, "rebar")
        img_subdir = 'img'
        image_dir = os.path.join(dataset_dir, img_subdir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            # image_ids = os.walk(image_dir)[2]
            image_ids = []
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if file.startswith('bar') \
                        or file.startswith('newset'):
                            name, ext = os.path.splitext(file)
                            image_ids.append(name)

            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))
                # mixed bar and newset
                newset_image_ids = [
                    image_id
                    for image_id in image_ids if image_id.startswith('newset')
                ]

                bar_image_ids = [
                    image_id
                    for image_id in image_ids if image_id.startswith('bar')
                ]

                np.random.shuffle(newset_image_ids)
                np.random.shuffle(bar_image_ids)

                newset_len = len(newset_image_ids)
                bar_len = len(bar_image_ids)

                image_ids = []
                image_ids.extend(bar_image_ids[0:int(bar_len*TRAIN_PORT)])
                image_ids.extend(newset_image_ids[0:int(newset_len*TRAIN_PORT)])

        # detect
        train_image_ids = set()
        valid_image_ids = set(image_ids)
        mask_output_dir = os.path.join(image_dir, '..', "mask_output")
        for root, dirs, files in os.walk(mask_output_dir):
            for file in files:
                name, ext = os.path.splitext(file)
                id = '_'.join(name.split('_')[:-1])
                if id in valid_image_ids and id not in train_image_ids:
                    train_image_ids.add(id)

        # Add images
        for image_id in train_image_ids:
            self.add_image(
                "rebar",
                image_id=image_id,
                path=os.path.join(image_dir, "{}.{}".format(
                    image_id, 'png' if image_id.startswith('bar') else 'jpg'
                ))
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "rebar":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(info['path']), '..' , "mask_output")

        # Read mask files from png image
        image_name = info['id']
        mask = []
        for root, dirs, files in os.walk(mask_dir):
            for f in files:
                if f.startswith(image_name):
                    m = skimage.io.imread(os.path.join(mask_dir, f)).astype(bool)
                    mask.append(m)

        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "rebar":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = RebarDataset()
    dataset_train.load_rebar(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RebarDataset()
    dataset_val.load_rebar(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.Sometimes(
        0.9,
        iaa.Sequential([
            iaa.OneOf([
                iaa.Fliplr(0.5), # horizontal flips
                iaa.Flipud(0.5), # horizontal flips
                ]
            ),
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.Sometimes(
                0.5,
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
            ),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Sometimes(
                0.5,
                iaa.Multiply((0.8, 1.2), per_channel=0.2)
            ),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.3, 1.7), "y": (0.3, 1.7)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-90, 90),
                shear=(-45, 45)
            )
        ], random_order=True) # apply augmenters in random order
    )

    # *** This training schedule is an example. Update to your needs ***
    # Training - Stage 1
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=80,
                augmentation=augmentation,
                layers='heads')

    # Training - Stage 2
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=200,
                layers='all',
                augmentation=augmentation)


############################################################
#  Detection
############################################################
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    height, width = image.shape[0], image.shape[1]
    red = [ [ [255, 0, 0] for j in range(width) ] for i in range(height) ]

    print("mask instance : {}".format( mask.shape[-1]))

    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) == 0)
        splash = np.where(mask, image, red).astype(np.uint8)
    else:
        splash = image

    return splash


def aug_image(image, gray):
    seq = iaa.Sometimes(
        0.9,
        iaa.Sequential([
            iaa.OneOf([
                iaa.Fliplr(0.5), # horizontal flips
                iaa.Flipud(0.5), # horizontal flips
                ]
            ),
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.Sometimes(
                0.5,
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
            ),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Sometimes(
                0.5,
                iaa.Multiply((0.8, 1.2), per_channel=0.2)
            ),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.3, 1.7), "y": (0.3, 1.7)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-90, 90),
                shear=(-45, 45)
            )
        ], random_order=True) # apply augmenters in random order
    )

    det = seq.to_deterministic()
    image = det.augment_image(image)
    gray = det.augment_image(gray)

    return image, gray


def test_mask(image_id):
    mask_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets', 'train', 'mask_output')
    # Read mask files from png image
    image_name = image_id
    mask = []
    print(mask_dir)
    for root, dirs, files in os.walk(mask_dir):
        for f in files:
            if f.startswith(image_name):
                print(f)
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(bool)
                # print(m.shape)
                mask.append(m)

    # print(mask.shape)
    mask = np.stack(mask, axis=-1)
    # Return mask, and array of class IDs of each instance. Since we have
    # one class ID, we return an array of ones

    augmentation = iaa.SomeOf((0, 3), [
        iaa.OneOf(
            [
                iaa.Affine(rotate=10),
                iaa.Affine(rotate=20),
                iaa.Affine(rotate=30),
                iaa.Affine(rotate=40),
                iaa.Affine(rotate=50),
                iaa.Affine(rotate=60),
                iaa.Affine(rotate=70),
                iaa.Affine(rotate=80),
            ]),
        iaa.Affine(
            scale={
                "x": (0.5, 1.5),
                "y": (0.9, 1.0)
            }
        ),
        iaa.Affine(
            scale={
                "x": (0.9, 1.0),
                "y": (0.5, 1.5)
            }
        ),
    ])

    image = skimage.io.imread(os.path.join(mask_dir, '..', 'img', '{}.jpg'.format(image_name)))

    # Make augmenters deterministic to apply similarly to images and masks
    det = augmentation.to_deterministic()
    image = det.augment_image(image)
    file_name = "splash_{0}_{1}.png".format(image_name, 0)
    skimage.io.imsave(file_name, image)

    # mask = det.augment_image(mask.astype(np.uint8))
    ins_count = mask.shape[-1]
    for i in range(ins_count):
        mask[:,:,i] = det.augment_image(mask[:,:,i].astype(np.uint8))

    instance_count = mask.shape[-1]
    for i in range(instance_count):
        img = mask[:,:,i]
        # img=det.augment_image(img)
        if np.any(img):
            file_name = "splash_{0}_{1}.png".format(image_name, i + 1)
            skimage.io.imsave(file_name, img)
        else:
            print('{} is all zero'.format(i))


def load_image(image_path):
    # Load image
    img = skimage.io.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if img.ndim != 3:
        img = skimage.color.gray2rgb(img)
    # If has an alpha channel, remove it for consistency
    if img.shape[-1] == 4:
        img = skimage.color.rgba2rbg(img)

    # conver to gray
    gray_img = np.uint8(skimage.color.gray2rgb(skimage.color.rgb2gray(img)) * 255)
    return img, gray_img


def detect_and_color_splash(model, image_path=None, aug=False):
    assert image_path

    # Image
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image, gray_image = load_image(image_path)

        if aug:
            image, gray_image = aug_image(image, gray_image)

        # Detect objects
        r = model.detect([gray_image], verbose=1)[0]

        # Color splash
        splash = color_splash(image, r['masks'])
        identify = '{:%Y%m%dT%H%M%S}'.format(datetime.datetime.now())

        file_name = "splash_{}_original.png".format(identify)
        skimage.io.imsave(file_name, image)

        file_name = "splash_{}_gray.png".format(identify)
        skimage.io.imsave(file_name, gray_image)

        file_name = "splash_{}.png".format(identify)
        skimage.io.imsave(file_name, splash)

        print("Saved to ", file_name)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = RebarConfig()
    else:
        config = RebarInferenceConfig()
    # config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, 'train')
    elif args.command == "detect":
        detect_and_color_splash(model, image_path=args.image)
        # test_mask('newset2_00193')
    elif args.command == "aug":
        detect_and_color_splash(model, image_path=args.image, aug=True)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
