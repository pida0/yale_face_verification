# yale_face_verification
Using pretrained alxnet on yale dataset   


## Dependencies
- pytorch 1.3.0
- tensorboard
- anaconda3 (python 3.7.4)  


## Dataset
[Yale face database](URL 'http://cvc.cs.yale.edu/cvc/projects/yalefaces/yalefaces.html')
(size 6.4MB) contains 165 grayscale images in GIF format of 15 individuals. There are 11 images per subject, one per different facial expression or configuration: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad, sleepy, surprised, and wink.   


## Note
- cpu only
- 8 images for training, 3 images for validation
- usage: `python train.py`  


## Result
25 epochs  
Training complete in 5m 15s  
Best val Acc: 1.000000  
![train acc](https://github.com/pida0/yale_face_verification/blob/master/result%20pic/train%20acc.png)
![train loss]()
![val acc]()
![val loss]()
