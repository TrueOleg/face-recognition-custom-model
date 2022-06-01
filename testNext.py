from tensorflow.keras.layers import Lambda, Layer


class QuadrupletLossLayer(Layer):
    '''
    A custom tf.keras layer that computes the quadruplet loss from distances
    ap_dist, an_dist, and nn_dist.
    The computed loss is independant from the batch size.

    Arguments:
        alpha, beta: margin factors used in the loss formula.
        inputs: (ap_dist, an_dist, nn_dist) with:
            ap_dist: distance between the anchor image (A) and the positive
                image (P) (of the same class),
            an_dist: distance between the anchor image (A) and the first image
                of a different class (N1),
            nn_dist: distance between the two images from different classes N1
                and N2 (that do not belong to the anchor class).

    External Arguments:
        LooksLikeWho.SLD_models.BATCH_SIZE: batch size used for training

    Output:
        The quadruplet loss per sample (averaged over one batch).
    '''

    def __init__(self, alpha, beta, **kwargs):
        self.alpha = alpha
        self.beta = beta

        super(QuadrupletLossLayer, self).__init__(**kwargs)

    def quadruplet_loss(self, inputs):
        ap_dist,an_dist,nn_dist = inputs

        #square
        ap_dist2 = K.square(ap_dist)
        an_dist2 = K.square(an_dist)
        nn_dist2 = K.square(nn_dist)

        return (K.sum(K.maximum(ap_dist2 - an_dist2 + self.alpha, 0), axis=0) + K.sum(K.maximum(ap_dist2 - nn_dist2 + self.beta, 0), axis=0)) / BATCH_SIZE

    def call(self, inputs):
        loss = self.quadruplet_loss(inputs)
        self.add_loss(loss)
        return loss

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
        })
        return config

