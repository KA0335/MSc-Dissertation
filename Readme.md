# Abstract
Generative Adversarial Networks (GANs) have become an active field for research as it can generate realistic images while learning from datasets which have no bounds. Often the latent space of a GAN is overlooked, and more emphasis is given to the model’s parameters and the dataset itself. Exploring the latent space of a GAN can not only shine some light on how to achieve better outputs from the model but also can help to produce on-demand variations of the image generated. Interpolation of latent points and efficient visualisation of the transition and feature arithmetic can help know that space better. This can lead to a better quality of synthetic image data which can include several edge cases which can be crucial while training a model or can serve as an inspiration by producing on-demand image variations.


--------------------------------------------------------------------------------------------------------------------


To achieve the exploration of the latent space and its visualisation, tsne plot was created of the latnet space, followed by feature addition, multiplication, etc. Also interpolation was done between different points. All these ideas are wrapped in an easy to use flask based local web interface. 

