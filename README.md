# advdecorr

[![Build Status](https://travis-ci.org/daniloefl/advdecorr.svg?branch=master)](https://travis-ci.org/daniloefl/advdecorr)

This project is a test bench for classification studies, where a classifier is expected to separate signal against a background without discriminating
on a variable S, referred to as `systematic uncertainty` throughout the code.

The training code is in the directory `training`, while a production web app and a server-side RESTful code are in the directories `app` and `restful`.
`docker-compose` can be used to create a docker image for both the web app and the server-side RESTful code using the `docker-compose.yml` configuration.
See more information about this in the next section.

Looking in the `training` directory, The training of the classifier can be done using the code in `keras_gan.py`,
where a GAN topology is used to penalize the classifier (taken as the GAN's generator)
if it is possible to learn the variable S. Another topology is implemented in `keras_aae.py`, where an adversarial auto-encoder is trained to censor
a latent representation, before using such latent representation for classification.

The data used for tests can be a real data sample or a toy data can be produced with `generateToys.py`.

An example of the results for a toy model can be seen in `training/result_toy`.

![Network output without adversary training](training/result_toy/ganna_discriminator_output_syst.pdf "Neural network output, trained without adversary")
![Network output with adversary training](training/result_toy/gan_discriminator_output_syst.pdf "Neural network output, trained with adversary")

# Flask client-server model test for production

The code in `production/` implements the production code, which receives data, loads the network and performs classification with the unseen input.
It can be run as a Flask REST API server with:

```
python classify.py
```

By default it runs in port 5001 in the local machine.

This can be transformed in a docker image with:

```
docker build -t advdecorr-restful .
docker run -d -p 5001:5001 advdecorr-restful
```

Now connecting to port 5001 outside of the container would work, but if the argument `-p 5001:5001` is omitted, this would not be possible in the host,
but it would be possible from other containers, which could be beneficial for isolation between the RESTful server and the Web application.

The code in `app/` implements the client-side web application, which communicates with the server-side to submit information and obtain new results.
It can be run as a Flask application with:

```
python web.py
```

It runs in port 5000 by default. Open the browser and navigate to `localhost:5000` and try to enter new data. After submitting, the client
communicates with the server and collects the p-values back, which are then shown in the resulting form.

The web application can be transformed in a docker image with:

```
docker build -t advdecorr-app .
docker run -d -p 5000:5000 advdecorr-app
```

The two docker containers can be built and ran using docker-compose, so that they are able to communicate with each other.
It can be done as follows, from the root directory of the project:

```
docker-compose up --build
```

