import os
import skimage

import pandas as pd
import numpy as np

from optparse import OptionParser

def main():
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir")
    parser.add_option("-o", "--output_dir", dest="output_dir")

    (options, args) = parser.parse_args()

    mask_dir = options.dir
    output_dir = options.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Read mask files from .png image
    mask = []
    mask_img = []

    for root, dirs, files in os.walk(mask_dir):
        for file in files:
            if file.endswith('.png'):
                root, ext = os.path.splitext(file)
                mask_img.append(root)

    for img in mask_img:
        m = skimage.io.imread(os.path.join(mask_dir, '{filename}.png'.format(filename=img)))
        # print(m.shape)
        # detect color instances
        # ignore background RGB: [0, 0, 0]
        instance_count = 0
        color_to_id = {}
        id_to_color = {}

        row = m.shape[0]
        col = m.shape[1]

        background = (0, 0, 0)

        for i in range(row):
            for j in range(col):
                r, g, b = m[i, j, 0], m[i, j, 1], m[i, j, 2]
                if (r, g, b) != background:
                    if (r, g, b) not in color_to_id:
                        id_to_color[instance_count] = (r,g,b)
                        color_to_id[(r,g,b)] = instance_count
                        instance_count += 1

        print(instance_count)

        # create output arr
        output_arr = np.zeros((row, col, instance_count), dtype=np.uint8)

        for i in range(row):
            for j in range(col):
                r, g, b = m[i, j, 0], m[i, j, 1], m[i, j, 2]
                if (r, g, b) != background:
                    id = color_to_id[(r,g,b)]
                    output_arr[i, j, id] = 1

        for id in range(instance_count):
            img_arr = np.zeros((row, col), dtype=np.uint8)
            for i in range(row):
                for j in range(col):
                    img_arr[i, j] = 255 if output_arr[i, j, id] != 0 else 0
            
            output_name = os.path.join(output_dir, '{filename}_{id}.png'.format(filename=img, id=id))
            skimage.io.imsave(output_name, img_arr)

if __name__ == '__main__':
    main()
