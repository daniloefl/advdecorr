#!/bin/sh

./generateToys.py
./keras_gan.py --input-file input_toys.h5 --prefix gan --train --load-trained 0 --iterations 51 --lr 1e-4
./keras_gan.py --input-file input_toys.h5 --prefix ganna --no-adv --train --load-trained 0 --iterations 51 --lr 1e-4

./keras_gan.py --input-file input_toys.h5 --prefix gan --plot-loss --load-trained 50
./keras_gan.py --input-file input_toys.h5 --prefix ganna --no-adv --plot-loss --load-trained 50

./keras_gan.py --input-file input_toys.h5 --prefix gan --plot-output --load-trained 50
./keras_gan.py --input-file input_toys.h5 --prefix ganna --no-adv --plot-output --load-trained 50

