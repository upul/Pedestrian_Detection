import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re

FILE_EXTRACTION_RE = re.compile(r'([a-zA-Z].*png).*\(([0-9]{1,3}),\s+([0-9]{1,3}),\s+([0-9]{1,3}),\s+([0-9]{1,3})\)')
GRIDS_PER_DIR = 8


def do_overlap(grid, box):
    """

    :param grid:
    :param box:
    :return:
    """
    if grid[1][0] < box[0][0] or grid[0][0] > box[1][0]:
        return False
    if grid[1][1] < box[0][1] or grid[0][1] > box[1][1]:
        return False
    return True


def get_grids(img):
    assert isinstance(img, np.ndarray)
    data = img[:, :, 0]
    size_x = data.shape[1]
    size_y = data.shape[0]
    n_blocks_x_dir = size_x // GRIDS_PER_DIR
    n_blocks_y_dir = size_y // GRIDS_PER_DIR

    all_grids = []
    for i in range(GRIDS_PER_DIR + 1):  # X dir
        for j in range(GRIDS_PER_DIR + 1):  # Y dir
            top_left = (j * n_blocks_y_dir, i * n_blocks_x_dir)
            bottom_right = (j * n_blocks_y_dir + n_blocks_y_dir, i * n_blocks_x_dir + n_blocks_x_dir)
            all_grids.append((top_left, bottom_right))
    return all_grids


def generate_label_data(images, bb_collection):
    """

    :param images: list of (image_file, numpy array)
    :param bb_collection: list of (image_file, [(top_left, bottom_right)]
    :return:
    """
    label_data = {}
    for img_file, img_data in images.items():
        image_size = img_data[:, :, 0]
        grid_width_x = image_size.shape[1] // GRIDS_PER_DIR
        grid_width_y = image_size.shape[0] // GRIDS_PER_DIR

        label = np.zeros((GRIDS_PER_DIR, GRIDS_PER_DIR))
        grids = get_grids(img_data)
        b_boxes = bb_collection[img_file]
        for box in b_boxes:
            for grid in grids:
                if do_overlap(grid, box):
                    i = grid[0][1] // grid_width_y  # Y dir
                    j = grid[0][0] // grid_width_x  # X dir
                    label[i, j] = 1
        label_data[img_file] = (grids, label)
    return label_data


def read_bounding_boxes(bb_files):
    bboxes = {}
    for file in bb_files:
        with open(file) as bb_file:
            for record in bb_file:
                selected = FILE_EXTRACTION_RE.search(record).groups()
                assert len(selected) == 5
                if selected[0] is None or selected[1] is None: # TODO (upul) check all
                    msg = 'Record should contain both file name and bounding boxes'
                    raise RuntimeError(msg)

                top_left = (int(selected[1]), int(selected[2]))
                bottom_right = (int(selected[3]), int(selected[4]))
                if record in bboxes:
                    bboxes[selected[0].strip()].append((top_left, bottom_right))
                else:
                    bboxes[selected[0].strip()] = [(top_left, bottom_right)]
    return bboxes


def read_images(image_dirs, extension='png'):
    """

    :param image_dirs: list of directories which contains images
    :return: list of tuples, each tuple contains (image_file, numpy array)
    """
    images = {}
    for dir in image_dirs:
        files = os.listdir(dir)
        for file in files:
            if extension in file:
                images[file] = plt.imread(os.path.join(dir, file))
    return images


def plot(image, grids):
    for grid in grids:
        cv2.rectangle(image, grid[0], grid[1], color=(200, 10, 2), thickness=1)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    image_dirs = ['./data/train-210']
    annotation_files = ['./data/train-210/annotation.txt']
    images = read_images(image_dirs)
    bboxes = read_bounding_boxes(annotation_files)
    print(bboxes['IFWN-sequence-048.avi-00053-R200.png'])
    labels = generate_label_data(images, bboxes)
    grids = labels['IFWN-sequence-048.avi-00053-R200.png'][0]
    plot(images['IFWN-sequence-048.avi-00053-R200.png'], grids)

