from sklearn.decomposition import PCA
from datatools.features import _get_video_data, VIDEO_WINDOW_SIZE, VIDEO_PRINCIPLE_COMPONENTS
import numpy as np
import cv2

input_files = ['./inputs/test01.txt',
               './inputs/test02.txt',
               './inputs/test03.txt',
               './inputs/test04.txt',
               './inputs/test05.txt',
               './inputs/test06.txt',
               './inputs/test07.txt',
               './inputs/test08.txt',
               './inputs/test09.txt',
               './inputs/test10.txt',
               './inputs/training_data.txt',
               ]

video_files = ['./video/test01.mp4',
               './video/test02.mp4',
               './video/test03.mp4',
               './video/test04.mp4',
               './video/test05.mp4',
               './video/test06.mp4',
               './video/test07.mp4',
               './video/test08.mp4',
               './video/test09.mp4',
               './video/test10.mp4',
               './video/training_data.mp4',
               ]

frames = []
for input_file, video_file in zip(input_files, video_files):
    _, video = _get_video_data(input_file,video_file, VIDEO_WINDOW_SIZE)
    frames.append(video)

pca = PCA(VIDEO_PRINCIPLE_COMPONENTS, random_state=42)
pca.fit(np.row_stack(frames))

frame01 = frames[0]

for i in xrange(30,35):
    out_str = '%02i' % i
    input_frame = np.reshape(frame01[i,:],(1,-1))
    original_image = np.reshape(input_frame,(VIDEO_WINDOW_SIZE, VIDEO_WINDOW_SIZE))
    new_image = np.reshape(pca.inverse_transform(pca.transform(input_frame)),(VIDEO_WINDOW_SIZE, VIDEO_WINDOW_SIZE))
    cv2.imwrite('./figures/features/original_' + out_str + '.png', original_image)
    cv2.imwrite('./figures/features/pca_' + out_str + '.png', new_image)