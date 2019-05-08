# advdecorr

This project is a test bench for classification studies, where a classifier is expected to separate signal against a background without discriminating
on a variable S, referred to as `systematic uncertainty` throughout the code.

The training of the classifier can be done using the code in `keras_gan.py`, where a GAN topology is used to penalize the classifier (taken as the GAN's generator)
if it is possible to learn the variable S. Another topology is implemented in `keras_aae.py`, where an adversarial auto-encoder is trained to censor
a latent representation, before using such latent representation for classification.

The data used for tests can be a real data sample or a toy data can be produced with `generateToys.py`.

# Flask client-server model test for production

The code in `production/` implements the production code, which receives data, loads the network and performs classification with the unseen input.
It can be run as a Flask REST API server with:

```
python classify.py
```

By default it runs in port 5001 in the local machine.

The code in `app/` implements the client-side web application, which communicates with the server-side to submit information and obtain new results.
It can be run as a Flask application with:

```
python web.py
```

It runs in port 5000 by default. Open the browser and navigate to `localhost:5000` and try to enter new data. After submitting, the client
communicates with the server and collects the p-values back, which are then shown in the resulting form.

