# Day_03_07_augmentation.py
import Augmentor


def usage_1():
    p = Augmentor.Pipeline('cifar10_mini')

    p.rotate(probability=0.7,
             max_left_rotation=10,
             max_right_rotation=10)
    p.zoom(probability=0.5,
           min_factor=1.1,
           max_factor=1.5)

    p.sample(3)


def usage_2():
    p = Augmentor.Pipeline('cifar10_mini')

    p.resize(probability=1, width=120, height=120)
    p.random_distortion(probability=1,
                        grid_width=4,
                        grid_height=4,
                        magnitude=2)

    p.sample(3)


# usage_1()
usage_2()




