# DWT + SVD Watermark Embedding System

This is an implementation of the DWT + SVD based Watermark Embedding System which uses SVM for watermark detection

### PROJECT TODOS: 

WaterMark Embedding:
- [x] Make a module to load and perform DWT of Image with coefficient extractino
- [x] Generate Watermark module to the method mentioned in the paper
- [x] Create Scheme to Embed Watermark in the coefficients 
- [x] Perform the Singular Value Decomposition to obtain the final watermarked Image


WaterMark Detection: 

- [x] Create a usable dataset with marked and unmarked images
- [x] Create a new SVM based classifier to mark the images
- [x] Test the trained model

Final Results: 

- [x] Calculate PSNR and NCC values and final test results of the prediction model 

Table: PSNR and NCC values for 166 sample images

| Metric          | Minimum | Maximum | Mean    | Standard Deviation |
|-----------------|---------|---------|---------|--------------------|
| PSNR            | 24.19   | 33.93   | 29.45   | 2.11               |
| NCC             | 0.94    | 0.99    | 0.99    | 0.01               |





