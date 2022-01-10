# GAN Image to Image Translation

**Note:** Work in progress, please scroll down to next steps section to see what is left. 

This project uses deep learning for image to image translation (changing one type of image into another type of image, for example an apple to an orange). Two state of the art Generative Adversarial Network (GAN) approaches were followed and implemented from scratch. 

**Please scroll down to the *screenshots* section to see preliminary examples of apples to oranges**

***

## Motivation
After seeing my brother draw animations frame by frame, I wondered if there was a more efficient way to get a character in different poses that was more time efficient. After researching a few deep learning approaches, I decided that using a GAN for image to image translation was possible. My idea is to take a 'blank character' (no details, only body outline/pose) and transform it into the detailed character in that pose. This is the final goal of the project: take an image of a pose and translate it to a character in that pose.  

***

## Project Structure

### data folder
Folder which contains the training datasets for the project. The data folder contains dataset folders in the form x_test, x_train, y_test, and y_train to create the dataloaders. 

### output folder
Folder which contains the GAN output for each epoch of training. 

### create_gen_discr.py
Contains custom Generator, Discriminator and Feature extractor networks (PyTorch modules) which use Residual blocks and a custom convolutional layer function (all in this file). 

### cut_model.py
Implementation from scratch of CUT GAN model from the [CUT GAN research paper](https://arxiv.org/pdf/2007.15651.pdf). 

### cycle_gan.py
Implementation from scratch of Cycle GAN model from the [Cycle GAN research paper](https://arxiv.org/pdf/1703.10593.pdf). 

### data_utility.py
Helper functions to create the dataloaders, save images, show batches, report losses, and set gradient status for networks. 

### losses.py
Implementation of discriminator GAN loss function module as well as PatchNCELoss module. 

### requirements.txt
Requirement file to be updated in the future for cloning the repo into a Python environment. 

### testing.py
Tests of network components to check functionality as they were being build from scratch. 

### train_gan.py
Training file for choosing model, setting hyperparameters, loading data, and training a GAN for image to image translation. 

***

## Tech/framework
**Built with**
- Pytorch (Python, deep learning)
- Matplotlib (Python, viewing images)
- TQDM (Python, showing training progress)

- **Note:** All of the modules were built from scratch in PyTorch so it was the only ML library used


## Project Checkpoints
1. Read [CUT GAN paper](https://arxiv.org/pdf/2007.15651.pdf). 
2. Create data utility functions (data loaders and viewing images). 
3. Implement code from scratch for the CUT GAN approach.
4. Train the CUT GAN model for apples -> oranges on a GPU only to find training issue
5. Pivot the approach to Cycle GAN
6. Read [Cycle GAN paper](https://arxiv.org/pdf/1703.10593.pdf). 
7. Implement code from scratch for Cycle GAN. 
8. Train and tweak Cycle GAN hyperparameters for training apples -> oranges (seems to be a more promising approach)

***

## Screenshots
Below is a preliminary result of the training (as this project is not fully complete). Go to the **output** folder for more results.  

![Apples to Oranges Sample](https://github.com/Nick-palmar/gan_img_to_img_translation/blob/main/output/ep37.png "Apples to Oranges Sample")




## Next Steps
1. Complete tweaking hyperparameters from apple -> orange with Cycle GAN
2. Try training a GAN model for zebra -> horse using the tweaked Cycle GAN hyperparameters.
3. Experiment with photograph -> painting in the style of a specific artist. 
4. Having a final Cycle GAN approach, use this to convert character pose -> animated character. 