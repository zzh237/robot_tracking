import numpy as np
import cv2
from sklearn.decomposition import PCA

VIDEO_WINDOW_SIZE = 40
VIDEO_PRINCIPLE_COMPONENTS = 20


def add_velocities(input_file, output_file):
    xy_pairs = []
    with open(input_file) as inf:
        for line in inf: xy_pairs.append([int(line.split(',')[0]), int(line.split(',')[1].strip())])

    new_values = []
    for i, xy_pair in enumerate(xy_pairs):
        x, y = xy_pair[0], xy_pair[1]
        # first point velocity is equal to the second point
        if i == 0:
            v_x = xy_pairs[i + 1][0] - xy_pairs[i][0]
            v_y = xy_pairs[i + 1][1] - xy_pairs[i][1]
            new_values.append([x, y, v_x, v_y])
        # other points velocity is equal to current location minus previous
        else:
            v_x = xy_pairs[i][0] - xy_pairs[i - 1][0]
            v_y = xy_pairs[i][1] - xy_pairs[i - 1][1]
            new_values.append([x, y, v_x, v_y])

    with open(output_file, 'w+') as outf:
        for robot_location_data in new_values:
            outf.writelines(','.join([str(point) for point in robot_location_data]) + '\n')


def add_video(input_files, output_files, video_files):

    number_data = []
    video_data = []

    for input_file, input_video in zip(input_files, video_files):
        numbers, video = _get_video_data(input_file,input_video, VIDEO_WINDOW_SIZE)
        video_data.append(video)
        number_data.append(numbers)

    pca = PCA(VIDEO_PRINCIPLE_COMPONENTS, random_state=42)
    pca.fit(np.row_stack(tuple(video_data)))

    for numbers, video, output_file in zip(number_data,video_data,output_files):
        transformed_video_data = pca.transform(video)
        out_data = np.column_stack((numbers,transformed_video_data))
        np.savetxt(output_file, out_data,'%.1f',',')


def _get_video_data(input_file,input_video, w):

    # Load in Data
    data = np.loadtxt(input_file, delimiter=",").astype('int64')

    # Load Video
    video = cv2.VideoCapture(input_video)

    # Loop over frames
    images = np.zeros((data.shape[0], w * w))
    for i in xrange(0, data.shape[0]):
        okay,frame = video.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = frame[(data[i, 1] - w // 2):(data[i, 1] + w // 2), (data[i, 0] - w // 2):(data[i, 0] + w // 2)]
        images[i,:] =np.reshape(frame,-1)

    return data, images


