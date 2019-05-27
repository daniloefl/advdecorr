#!/bin/sh

./generateToys.py
./keras_gan.py --input-file input_toys.h5 --prefix gan --train --iterations 101
./keras_gan.py --input-file input_toys.h5 --prefix ganna --no-adv --train --iterations 101

./keras_gan.py --input-file input_toys.h5 --prefix gan --plot-loss --load-trained 100
./keras_gan.py --input-file input_toys.h5 --prefix ganna --no-adv --plot-loss --load-trained 100

./keras_gan.py --input-file input_toys.h5 --prefix gan --plot-output --load-trained 100
./keras_gan.py --input-file input_toys.h5 --prefix ganna --no-adv --plot-output --load-trained 100

