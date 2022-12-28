# Abstract
Generative Adversarial Networks (GANs) have become an active field for research as it can generate realistic images while learning from datasets which have no bounds. Often the latent space of a GAN is overlooked, and more emphasis is given to the model’s parameters and the dataset itself. Exploring the latent space of a GAN can not only shine some light on how to achieve better outputs from the model but also can help to produce on-demand variations of the image generated. Interpolation of latent points and efficient visualisation of the transition and feature arithmetic can help know that space better. This can lead to a better quality of synthetic image data which can include several edge cases which can be crucial while training a model or can serve as an inspiration by producing on-demand image variations.


--------------------------------------------------------------------------------------------------------------------


To achieve the exploration of the latent space and its visualisation, tsne plot was created of the latnet space, followed by feature addition, multiplication, etc. Also interpolation was done between different points. All these ideas are wrapped in an easy to use flask based local web interface. 


![img](https://github.com/KA0335/MSc-Dissertation/blob/main/images/Screenshot%20(104).png)
This shows the t-sne plot used to get an estimate on how the images are related to each other based on their features.


![img](https://github.com/KA0335/MSc-Dissertation/blob/main/images/Screenshot%20(105).png)
Then all the images are out on display for the user to see them clearly. (the main aim of this study was to examine the latent space and not generate quality images from a GAN)


![img](https://github.com/KA0335/MSc-Dissertation/blob/main/images/Screenshot%20(113).png)
 
 The user could input imgae numbers to see the model in action.


![img](https://github.com/KA0335/MSc-Dissertation/blob/main/images/Screenshot%20(112).png)
Finally there is a new page where the results of interpolation, and feature addition are displayed.

------------------------------------------------------------------------------------------------------------------

The H5.py python files iterates through a large number of checkpoints that and then the best quality h5 file is selected for the above interface. For a deeper understanding of the model and access, please contact me via my email: anandkushagra2898@gmail.com

All credits to the respective owners who helped me build upon this project.
