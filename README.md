## reid_tracker
Zero-shot ReID baased human tracker for service robots. It can retrack missing human from the camera frame.  
It creates human identity gallary then do a ranking based system.  
It directly gets the camera devices since the sensor_msgs/Image transport on ROS2 is laggy and not viable like ROS1 transport.  

Body Feature Extraction:  
1. DINOv3 : Body Part Feature
2. SFace : Facial Feature  

The technical details are explained on this poster  

Please get the following ONNX File from the below link   

1. Yunet Face Detection : https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
2. SFace Facial Feature : https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx  
3. RTMO Pose Detection  :  https://drive.google.com/file/d/1grya1leJeUd-GtQAq3DcDIU8RHts2TSq/view?usp=sharing
4. RTDETRv1 : https://huggingface.co/onnx-community/rtdetr_r50vd  
5. DINOv3 : you can straightforward convert DINOv3 pt to onnx. get DINOV3 pt from facebook : ViT-B/16 distilled  

