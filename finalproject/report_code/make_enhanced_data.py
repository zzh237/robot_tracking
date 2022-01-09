from datatools.features import add_velocities, add_video

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


velocity_output_files = ['./inputs/test01_with_velocity.txt',
                         './inputs/test02_with_velocity.txt',
                         './inputs/test03_with_velocity.txt',
                         './inputs/test04_with_velocity.txt',
                         './inputs/test05_with_velocity.txt',
                         './inputs/test06_with_velocity.txt',
                         './inputs/test07_with_velocity.txt',
                         './inputs/test08_with_velocity.txt',
                         './inputs/test09_with_velocity.txt',
                         './inputs/test10_with_velocity.txt',
                         './inputs/training_data_with_velocity.txt',
                        ]

video_output_files = ['./inputs/test01_with_velocity_and_video.txt',
                      './inputs/test02_with_velocity_and_video.txt',
                      './inputs/test03_with_velocity_and_video.txt',
                      './inputs/test04_with_velocity_and_video.txt',
                      './inputs/test05_with_velocity_and_video.txt',
                      './inputs/test06_with_velocity_and_video.txt',
                      './inputs/test07_with_velocity_and_video.txt',
                      './inputs/test08_with_velocity_and_video.txt',
                      './inputs/test09_with_velocity_and_video.txt',
                      './inputs/test10_with_velocity_and_video.txt',
                      './inputs/training_data_with_velocity_and_video.txt',
                        ]

# Add velocity data to standard data files
for input_file, output_file in zip(input_files,velocity_output_files):
    add_velocities(input_file, output_file)

# Add video data to velocity files
add_video(velocity_output_files, video_output_files, video_files)

