#!/bin/sh

./generateToys.py
./keras_gan.py --input-file input_toys.h5 --prefix gan --mode train --iterations 101
./keras_gan.py --input-file input_toys.h5 --prefix ganna --no-adv --mode train --iterations 101

./keras_gan.py --input-file input_toys.h5 --prefix gan --mode plot_loss --load-trained 100
./keras_gan.py --input-file input_toys.h5 --prefix ganna --no-adv --mode plot_loss --load-trained 100

./keras_gan.py --input-file input_toys.h5 --prefix gan --mode plot_output --load-trained 100
./keras_gan.py --input-file input_toys.h5 --prefix ganna --no-adv --mode plot_output --load-trained 100

