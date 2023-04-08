## Get ready to script
brew install geos conda create -n histoTK python=3.9 conda activate histoTK pip install shapely==1.7.1

git clone <https://github.com/DigitalSlideArchive/HistomicsTK/> cd HistomicsTK python setup.py install

python setup.py build_ext --inplace conda install -n histoTK ipykernel --update-deps --force-reinstall

pip install torch torchvision segmentation-models-pytorch ipykernel matplotlib albumentations --force-reinstall --no-cache-dir pip install ipywidgets jupyter docker -U

conda install openslide

turn off the other kernelspec
jupyter kernelspec list -Xfrozen_modules=off

## Downloading the data
You can download the AIDPATH dataset from their website (<https://aidpath.org/data/>) or access it through their API. You will need to create an account to access the data.

## Preparing the data
Once you have downloaded the data, you will need to prepare it for training. As mentioned in the paper, the images and masks need to be paired and equisized. You will need to extract image patches and their corresponding masks from the WSIs.

## Data augmentation
To improve the generalization and robustness of the U-Net model, you will need to augment the data. This involves applying several random orientation perturbations (rotation and diagonal flip) followed by random stain color and contrast perturbations.

## Training the U-Net model
You will need to modify the basic U-Net model by extending the model depth from three to five levels and adding a dropout factor equal to 0.25. The dropout layer should be added after each convolution block. The U-Net should be trained for 10 epochs and the segmentation performance should be monitored after each epoch by the Dice coefficient on the validation set. The learning rate, dropout rate, and batch size for the stochastic gradient descent optimizer should be set to 0.01, 0.25, and 3, respectively.

## Testing the U-Net model
To test the U-Net model, you will need to apply a tissue masking algorithm to a gray level WSI image at very low magnification (2x) to isolate the tissue from white optical background. Subsequently, the receptive field of the trained U-Net model should be slid over the tissue image under the mask, with the sliding interval equal to 50% of the receptive field size (256 pixels). This approach causes some overlap between masks outputted by the U-Net. To determine the class label for pixels in the overlapping areas in the outputted masks, the class majority voting should be applied. After the sliding concludes, the masks outputted by U-Net should be stitched and detected candidate glomerular objects should be evaluated for size (post-processing) in the stitched image.

As for coding this, it's a bit of a larger task than I can write out here. However, I can give you some pointers on where to start. Here are some suggestions:

1. Start by downloading the AIDPATH dataset and loading some of the images and their corresponding masks into your environment.

2. Write a function that extracts image patches and their corresponding masks from the WSIs.

3. Write a function that applies random orientation perturbations (rotation and diagonal flip) followed by random stain color and contrast perturbations to the image patches and their masks.

4. Modify the basic U-Net model by extending the model depth from three to five levels and adding a dropout factor equal to 0.25. The dropout layer should be added after each convolution block.

5. Train the U-Net model using the prepared and augmented image patches and their corresponding masks.

6. Test the U-Net model by applying a tissue masking algorithm to a gray level WSI image at very low magnification (2x) to isolate the tissue from white optical background. Subsequently, slide the receptive field of the trained U-Net model over the tissue image under the mask, with the sliding interval equal to 50% of the receptive field size (256 pixels). Apply class majority voting to determine the class label for pixels in the overlapping areas in the outputted masks. Stitch the masks outputted by U-Net and evaluate detected candidate glomerular objects for size (post-processing) in the stitched image

docker-compose exec girder bash ls /home/
