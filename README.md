# otsuThresholding
This is a working implementation of the the otsu algorithm. It was originally part of a larger project but I decided to cut it out and put it on here. This part was originally part of a series of pre-edits wich make images easier to read for Google Tesseract.
This method will yield best results with Google Tesseract if it is combined with a Region-Of-Interest Analysis.

With this application images are binarized, meaning only 0 and 1 (black and white) remain. In order to achieve this I use OpenCV.
A simple sample image with a difficult case has already been added and will be binarized with the custom otsu algorithm.

Here is an example to showcase how an image is turned into a binary image:
![otsu thresholding example](https://i.imgur.com/oA3LQ5n.png)
