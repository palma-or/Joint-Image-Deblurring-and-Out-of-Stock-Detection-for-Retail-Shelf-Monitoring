# Joint Image Deblurring and Out-of-Stock Detection for Retail Shelf Monitoring

<img width="2784" height="1536" alt="Panoramica" src="https://github.com/user-attachments/assets/fa6d2857-3044-428d-af90-d15fd5cc8709" />

This repository is designed for studying and mitigating the impact of motion blur in mobile robot Out-of-Stock (OOS) detection. The large scale retail distribution sector has been facing the OOS problem for years, with a direct economic impact on the retailer in terms of lost revenue and customer dissatisfaction. The automation of monitoring through mobile robotic platforms, such as the Pepper robot, represents an effective solution but introduces a critical challenge: the motion blur introduced during image acquisition. This form of degradation is directional, spatially variable, and its intensity depends on the instantaneous speed of the robot and the depth of the framed scene.

This project proposes a joint pipeline of image deblurring and OOS detection to overcome this limitation. The approach uses a generative module to translate input images towards a known blur domain before they reach the detector.

## Contributions and Pipeline Architecture

⚠️ **Note:** This codebase introduces a systematic approach to out-of-distribution blur generalization without requiring real blurred video frames during training.

The system is structured and evaluated on three main pipeline variants:

* **Detector-Only (Baseline)**: The YOLO26 detector processes the image directly without any enhancement module upstream.
  <img width="2385" height="845" alt="fig17a_pipeline_overview_detector_only" src="https://github.com/user-attachments/assets/cffb609a-1689-4184-b8da-87d637b805d1" />

* **GAN Frozen Pipeline**: The pre-trained Blur2Blur generative module pre-processes the image with fixed weights before it passes to the detector.
  <img width="3780" height="845" alt="fig17b_pipeline_overview_gan_frozen" src="https://github.com/user-attachments/assets/edbd578a-1d58-4b3b-b2ac-91a6411fbb9a" />

* **GAN Joint Pipeline**: The generator's weights are updated jointly with the detector via backpropagation of the detection loss.
  <img width="3780" height="1066" alt="fig17c_pipeline_overview_joint" src="https://github.com/user-attachments/assets/385f9bdb-4d99-4ab8-92f3-cb337b696e5c" />

* **The Blur2Blur Module**: The system architecture relies on the Blur2Blur framework to translate images from an unknown blur domain to a known one. The construction of the dataset for training the module does not require aligned pairs of blurry and sharp images, making the framework completely unsupervised.
  
  <img width="477" height="292" alt="Blur2Blur1" src="https://github.com/user-attachments/assets/b0db3222-493b-42b7-b6ee-d5a65d253f43" />

## Contents

* [Installation](#installation)
* [Dataset Composition](#dataset-composition)
* [Getting Started](#getting-started)
* [Training](#training)
* [Evaluation & Visual Results](#evaluation--visual-results)
* [Citation](#citation)
* [License](#license)

## Installation

Please run the following commands in the given order to install the dependencies for this pipeline.

```bash
git clone [https://github.com/palma-or/oos_orlando.git](https://github.com/palma-or/oos_orlando.git)
cd oos_orlando
pip install -r requirements.txt
pip install torch ultralytics opencv-python numpy pillow albumentations matplotlib pyyaml
```

## Dataset Composition

The experiments were conducted using a multi-source dataset comprising 11,519 images. This dataset includes:
* Retail shelf images from public datasets (SKU-110K, Grocery Products, WebMarket, Roboflow) re-annotated specifically for the slot-level "empty" class.
* Private static datasets collected by the MIVIA group inside real supermarkets.
* The MIVIA Robotic OOS Dataset, composed exclusively of real video frames acquired by the Pepper robot during autonomous navigation, employed only as a test set.

To prepare the data for the GAN and the detector, utilize the provided scripts in the `GAN/code/` and `training/utils/` directories:

```bash
# 1. Analyze blur variance in video frames
python GAN/code/blur_analysis.py --frames_dir <frames> --output_dir <output>

# 2. Build datasets for the Blur2Blur module
python GAN_code/step1_build_dataset.py --frames_dir <frames> --output_dir <blur2blur_data> --threshold 200
python GAN_code/step2_build_k.py --data_dir <blur2blur_data> --blur2blur_dir <path/to/Blur2Blur>

# 3. Apply Pepper-calibrated synthetic motion blur to store images
python training/utils/augment_store.py --store-dir <Store> --output-dir <Store_augmented>
```

## Getting Started

For a detailed walk-through of the codebase structure, refer to the `training/code/` directory. The pipeline uses YOLO26n pre-trained weights (`yolo26n.pt`) and handles dataset configurations via YAML files (e.g., `data.yaml`, `data_store.yaml`) located in the `data/` folder.

## Training

To start an OOS detection training experiment, you can choose between the three pipeline modes. The training script uses Ultralytics callbacks to seamlessly integrate the GAN.

Run the following commands based on your target architecture:

```bash
# Baseline YOLO26n without GAN
python training/code/train_joint.py --mode detector_only --data data/data.yaml

# GAN pre-processes images, weights stay frozen
python training/code/train_joint.py --mode gan_frozen --data data/data.yaml --gan_ckpt GAN/checkpoints/ckpt.pth

# GAN weights are updated via detection loss after warm-up
python training/code/train_joint.py --mode joint --data data/data.yaml --gan_ckpt GAN/checkpoints/ckpt.pth --phase1_epochs 5
```

## Evaluation & Visual Results

The integration of the generative module effectively standardizes the visual style between the training images synthetically blurred and the test images with real blur introduced by the robot.

<img width="2687" height="2600" alt="fig18_train_test_gan_comparison" src="https://github.com/user-attachments/assets/9bcba968-00cd-4e1c-8533-4d49ce095123" />

The results demonstrate a marked improvement in the system's ability to correctly localize empty shelf slots compared to the baseline. The optimal combination that emerged from the experiments, fixed-weight generative module on a specific dataset, makes the collection of real videos from the target platform superfluous while still achieving solid performance.

<img width="4020" height="1335" alt="fig07_baseline_gan_frozen_joint" src="https://github.com/user-attachments/assets/ebd6ea44-3637-4f9f-a04e-13e48c6ef497" />

Furthermore, an original evaluation framework was developed that breaks down the error into a disjoint count of False Positives and False Negatives calculated on a systematic grid. This breakdown allows for a finer and more stable reading of the system's behavior than a single aggregated metric can offer.

<img width="2117" height="1517" alt="error_chart" src="https://github.com/user-attachments/assets/9a9b6119-c2d9-4ac1-bda3-36dfc5e34e6a" />

```bash
# Compare all three pipelines side by side
python training/code/visualize_pipelines.py \
    --images <images> --labels <labels> \
    --weights-detector <best_det.pt> --weights-frozen <best_froz.pt> --weights-joint <best_joint.pt> \
    --gan-ckpt <latest_net_G.pth> --blur2blur-dir <path/to/Blur2Blur> \
    --data-yaml data/data.yaml --output <output_dir>

# Categorize every False Positive/Negative by likely cause
python training/code/analyze_errors.py \
    --runs-dir runs/ --image-dir <images> --label-dir <labels> \
    --runs oos_gan_frozen oos_joint --output <output_dir>
```

## License

| Component | License |
| :--- | :--- |
| Codebase | MIT License |
| Datasets | Custom / Private (MIVIA) & Public Licenses |
