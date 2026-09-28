# Joint Image Deblurring and Out-of-Stock Detection for Retail Shelf Monitoring

<img width="2784" height="1536" alt="Panoramica" src="https://github.com/user-attachments/assets/fa6d2857-3044-428d-af90-d15fd5cc8709" />

## 📖 Problem Description
The large scale retail distribution sector has been facing the out-of-stock (OOS) problem for years, with a direct economic impact on the retailer in terms of lost revenue and customer dissatisfaction. The automation of monitoring through mobile robotic platforms, such as the Pepper robot, represents an effective solution but introduces a critical challenge: the motion blur introduced during image acquisition. This form of degradation is directional, spatially variable, and its intensity depends on the instantaneous speed of the robot and the depth of the framed scene.

This project proposes a joint pipeline of image deblurring and OOS detection to overcome this limitation. The approach uses a generative module to translate input images towards a known blur domain before they reach the detector.

## 🧠 Pipeline Architecture
The system is structured and evaluated on three main pipeline variants:

* **Detector-Only (Baseline)**: The YOLO26 detector processes the image directly without any enhancement module upstream.
<img width="2385" height="845" alt="fig17a_pipeline_overview_detector_only" src="https://github.com/user-attachments/assets/cffb609a-1689-4184-b8da-87d637b805d1" />

* **GAN Frozen**: The pre-trained Blur2Blur generative module pre-processes the image with fixed weights before it passes to the detector.
<img width="3780" height="845" alt="fig17b_pipeline_overview_gan_frozen" src="https://github.com/user-attachments/assets/edbd578a-1d58-4b3b-b2ac-91a6411fbb9a" />

* **GAN Joint**: The generator's weights are updated jointly with the detector via backpropagation of the detection loss.
<img width="3780" height="1066" alt="fig17c_pipeline_overview_joint" src="https://github.com/user-attachments/assets/385f9bdb-4d99-4ab8-92f3-cb337b696e5c" />

## 🛠 The Blur2Blur Module
The system architecture relies on the Blur2Blur framework to translate images from an unknown blur domain to a known one. The construction of the dataset for training the module does not require aligned pairs of blurry and sharp images, making the framework completely unsupervised.
<img width="477" height="292" alt="Blur2Blur1" src="https://github.com/user-attachments/assets/b0db3222-493b-42b7-b6ee-d5a65d253f43" />

## 📊 Dataset Composition
The experiments were conducted using a multi-source dataset comprising 11,519 images[cite: 4]. This dataset includes:
* Retail shelf images from public datasets (SKU-110K, Grocery Products, WebMarket, Roboflow) re-annotated specifically for the slot-level "empty" class.
* Private static datasets collected by the MIVIA group inside real supermarkets.
* The MIVIA Robotic OOS Dataset, composed exclusively of real video frames acquired by the Pepper robot during autonomous navigation, employed only as a test set.

## 📈 Visual Results and Error Analysis
The integration of the generative module effectively standardizes the visual style between the training images synthetically blurred and the test images with real blur introduced by the robot.
<img width="2687" height="2600" alt="fig18_train_test_gan_comparison" src="https://github.com/user-attachments/assets/9bcba968-00cd-4e1c-8533-4d49ce095123" />


The results demonstrate a marked improvement in the system's ability to correctly localize empty shelf slots compared to the baseline. The optimal combination that emerged from the experiments, fixed-weight generative module on a specific dataset, makes the collection of real videos from the target platform superfluous while still achieving solid performance.
<img width="4020" height="1335" alt="fig07_baseline_gan_frozen_joint" src="https://github.com/user-attachments/assets/ebd6ea44-3637-4f9f-a04e-13e48c6ef497" />


Furthermore, an original evaluation framework was developed that breaks down the error into a disjoint count of False Positives and False Negatives calculated on a systematic grid. This breakdown allows for a finer and more stable reading of the system's behavior than a single aggregated metric can offer.
<img width="2117" height="1517" alt="error_chart" src="https://github.com/user-attachments/assets/9a9b6119-c2d9-4ac1-bda3-36dfc5e34e6a" />
