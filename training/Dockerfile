FROM python:3

LABEL maintainer="daniloefl@gmail.com"

COPY ./keras_gan.py /training/keras_gan.py
COPY ./keras_aae.py /training/keras_aae.py
COPY ./generateToys.py /training/generateToys.py
WORKDIR /training

RUN pip install numpy h5py pandas tensorflow keras matplotlib

ENTRYPOINT [ "python" ]

CMD ["generateToys.py"]

CMD ["keras_gan.py", "--input-file", "input_toys.h5", "--prefix", "gan", "--mode", "train", "--iterations", "50"]
CMD ["keras_gan.py", "--input-file", "input_toys.h5", "--prefix", "ganna", "--no-adv", "--mode", "train", "--iterations", "50"]

CMD ["keras_gan.py", "--input-file", "input_toys.h5", "--prefix", "gan", "--mode", "plot_loss", "--load-trained", "50"]
CMD ["keras_gan.py", "--input-file", "input_toys.h5", "--prefix", "ganna", "--no-adv", "--mode", "plot_loss", "--load-trained", "50"]

CMD ["keras_gan.py", "--input-file", "input_toys.h5", "--prefix", "gan", "--mode", "plot_output", "--load-trained", "50"]
CMD ["keras_gan.py", "--input-file", "input_toys.h5", "--prefix", "ganna", "--no-adv", "--mode", "plot_output", "--load-trained", "50"]

