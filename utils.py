import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

GRIDS_PER_DIR = 8


def do_overlap(grid, box):
    """

    :param grid:
    :param box:
    :return:
    """
    top_left = grid[2]
    bot_right = grid[3]
    if bot_right[0] < box[0][0] or top_left[0] > box[1][0]:
        return False
    if bot_right[1] < box[0][1] or top_left[1] > box[1][1]:
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
    for i in range(GRIDS_PER_DIR):  # X dir
        for j in range(GRIDS_PER_DIR):  # Y dir
            top_left = (i * n_blocks_x_dir, j * n_blocks_y_dir)
            bottom_right = (i * n_blocks_x_dir + n_blocks_x_dir, j * n_blocks_y_dir + n_blocks_y_dir)
            all_grids.append((i, j, top_left, bottom_right))
    return all_grids


def generate_label_data(training_images, bb_collection):
    """

    :param training_images: list of (image_file, numpy array)
    :param bb_collection: list of (image_file, [(top_left, bottom_right)]
    :return:
    """
    label_data = {}
    for img_file, img_data in training_images.items():
        img_label = np.zeros((GRIDS_PER_DIR, GRIDS_PER_DIR))
        img_grids = get_grids(img_data)
        img_bounding_boxes = bb_collection[img_file]
        for box in img_bounding_boxes:
            for grid in img_grids:
                if do_overlap(grid, box):
                    i = grid[1]
                    j = grid[0]
                    img_label[i, j] = 1
        label_data[img_file] = (img_grids, img_label)
    return label_data


def read_bounding_boxes(bb_files):
    all_bounding_boxes = {}
    for file in bb_files:
        with open(file) as bb_file:
            for row in bb_file:
                if 'xmin' in row:
                    continue
                else:
                    cols = row.split(',')
                    assert len(cols) == 7
                    top_left = (int(cols[0]), int(cols[1]))
                    bot_right = (int(cols[2]), int(cols[3]))
                    frame = cols[4]
                if frame in all_bounding_boxes:
                    all_bounding_boxes[frame].append((top_left, bot_right))
                else:
                    all_bounding_boxes[frame] = [(top_left, bot_right)]
    return all_bounding_boxes


def read_images(image_dirs, ext='jpg'):
    """

    :param ext: 
    :param image_dirs: list of directories which contains images
    :return: list of tuples, each tuple contains (image_file, numpy array)
    """
    images = {}
    for dir in image_dirs:
        files = os.listdir(dir)
        for file in files:
            if ext in file:
                images[file] = plt.imread(os.path.join(dir, file))
    return images


def plot(image, grids, label, bbx):
    for grid in grids:
        cv2.rectangle(image, grid[2], grid[3], color=(200, 10, 2), thickness=1)

    x_size, y_size = image.shape[1] // GRIDS_PER_DIR, image.shape[0] // GRIDS_PER_DIR
    copy = image.copy()
    for j in range(GRIDS_PER_DIR):
        for i in range(GRIDS_PER_DIR):
            if label[i, j] == 1:
                cv2.rectangle(copy, (j * x_size, i * y_size), (j * x_size + x_size, i * y_size + y_size),
                              color=(200, 1, 1), thickness=-1)
    alpha = 0.2
    cv2.addWeighted(copy, alpha, image, 1 - alpha, 0, image)
    grid = get_grids(image)
    for g in grid:
        cv2.rectangle(image, g[2], g[3], color=[100, 100, 50], thickness=2)

    for bb in bbx:
        cv2.rectangle(image, bb[0], bb[1], color=[100, 50, 50], thickness=3)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    img_file_name = '1479498372942264998.jpg'
    image_dirs = ['/home/upul/datasets/udacity/debug']
    annotation_files = ['/home/upul/datasets/udacity/debug/labels.csv']
    images = read_images(image_dirs)
    bboxes = read_bounding_boxes(annotation_files)
    print(bboxes[img_file_name])
    labels = generate_label_data(images, bboxes)
    grids = labels[img_file_name][0]
    label = labels[img_file_name][1]
    print(label)
    plot(images[img_file_name], grids, label,
         bboxes[img_file_name])
