# wgan-and-wgan-gp-with-gluon
implement wgan and wgan-gp in gluon and compare the quality of generated images with inception score and FID(in tensorflow)


## generated images:

- with dcgan

![](https://github.com/veetsin/wgan-and-wgan-gp-with-gluon/blob/master/images/dcgan_mnist_fig.png)

- with wgan

![](https://github.com/veetsin/wgan-and-wgan-gp-with-gluon/blob/master/images/wgan_mnist_fig.png)


- with wgan-gp

![](https://github.com/veetsin/wgan-and-wgan-gp-with-gluon/blob/master/images/wgan-gp_mnist.png)


## plot of the wasserstein distance

![](https://github.com/veetsin/wgan-and-wgan-gp-with-gluon/blob/master/images/plot_wd.png)


##compare images with inception score and FID as metric 

![](https://github.com/veetsin/wgan-and-wgan-gp-with-gluon/blob/master/images/is_fid.png)


this part used code from official implementation with tensorflow

[inception-score](https://github.com/openai/improved-gan)
[FID](https://github.com/tsc2017/Frechet-Inception-Distance)
