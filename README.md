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

Table: PSNR and NCC values for 183 sample images

| Metric          | Minimum | Maximum | Mean    | Standard Deviation |
|-----------------|---------|---------|---------|--------------------|
| PSNR            | 28.1104 | 40.4185 | 35.1916 | 2.5539             |
| NCC             | 0.9856  | 0.9996  | 0.9961  | 0.0026             |




