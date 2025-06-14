# DP-AdaFit

DP-AdaFit is a training framework that enables high-quality generative modeling under differential privacy (DP) constraints. It leverages parameter-efficient AdaLoRA tuning with adaptive noise injection and proposes a **Redundant Noise Reduction (RNR)** mechanism to improve generation quality without compromising the formal privacy budget.

---

## 🔍Highlights

- ✅ Supports **differential privacy (ε, δ)**-bounded training for generative models.
- ✅ Integrates **AdaLoRA (Adaptive Low-Rank Adaptation)** for efficient and scalable fine-tuning.
- ✅ Proposes a novel **Redundant Noise Reduction (RNR)** mechanism to mitigate performance degradation due to DP noise.
- ✅ Offers plug-and-play support for guided-diffusion, stylegan2, stable diffusion, etc.
- ✅ Enables **privacy-utility trade-off** control via calibrated noise scheduling.
---

## 📁 Project Structure

```bash
├── checkpoints/             # checkpoint to save in training
├── datasets/                # preprossing the datasets
├── evaluations/             # FID / sFID / IS 
├── guided_diffusion/        # model and training code
├── pretrained_models/       # the directory of pretrained models
├── scripts/                 # Main entry for DP training, sample,
├── src/                     # the custom RNR and DP finetuning in AdaFit
└── README.md
```
## ⚙️ Installation
### Clone repository
```bash
git clone https://github.com/yeohhoo/DP-AdaFit
cd DP-AdaFit
```
### Create environment
```bash
conda create -n dp-adafit python=3.10
conda activate dp-adafit
```
### Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start
  ### Pretrained Model 
To reduce privacy cost, we first pretrain the guided-diffusion [model](https://drive.google.com/drive/folders/1gx7Vx7kvGa78taePSJ_wrkiyZrYPRwNI?usp=sharing) on the public ImageNet32 dataset, enabling the model to learn general features before fine-tuning on sensitive data. For more implementation details, please refer to the [guided-diffusion](https://github.com/openai/guided-diffusion) repository.

## Training

###  Preparing your Data
+ You should first download your private dataset and organize the file structure as follows:
```
└── data_root
    ├── train                      
    |   ├── cate-id                                   # train-class
    |   |    ├── cate-id_sample-id.jpg                # train-img
    |   |    └── ...                                  # ...
    |   └── ...                                       # ...
    └── valid                      
        ├── cate-id                                   # valid-class
        |    ├── cate-id_sample-id.jpg                # valid-img
        |    └── ...                                  # ...
        └── ...                                       # ...
```
* Here, we provide organized cifar10 dataset as an example:
```
└── data_root
    ├── train
    |   ├── bird
    |   |    ├── bird_00006.png
    |   |    └── ...
    |   └── ...
    └── valid
        ├── bird
        |    ├── bird_00025.png
        |    └── ...
        └── ...
```
### Training with differential privacy
 ```
 MODEL_FLAGS="--image_size 32 --num_channels 192 --num_res_blocks 2 --learn_sigma True --dropout 0.3" 
 ```
 ```
 DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear" 
 ```
 ```
 TRAIN_FLAGS="--lr 1e-3 --batch_size 16384 --num_steps 200 --train_path data_root/train --val_path data_root/test --resume_checkpoint pretrained_models/your_model_pub.pt --use_pretrain 1 --log_interval 5" 
 ```
 ```
 PRIVACY_FLAG="--epsilon 10 --delta 1e-5 --max_per_sample_grad_norm 1e-3 --timestep_mul 2 --transform 2 --max_physical_batch_size 320" 
 ```
 ```
 DIST_FLAGS="--n_gpus_per_node 1 --master_port 6887 --omp_n_threads 1" 
 ```
 ```
 LoRA_FLAG="--lora_r 9,9,9,9 --target_rank 6"
 ```
 ```
 PYTHONPATH=your_root/DP-AdaFit python scripts/image_train_aug.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $PRIVACY_FLAG $DIST_FLAGS $LoRA_FLAG
 ```
 Note: our code also surpport reseme training by setting the parameters `--resume_step` and `--resume_checkpoint`. For data loading, there is other variable `CACHE_DATASET` in `src/utils/folder.py` need to setting.

## Testing
Having trained your model or using pre-trained models we provide, you can use scripts/image_sample.py to apply the model to generate more samples.
### Sample
```
MODEL_FLAGS="--image_size 32 --num_channels 192 --num_res_blocks 2 --learn_sigma True --dropout 0.3" DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
```
```
 python scripts/image_sample.py --model_path pretrained_models/checkpoint_xxx.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```
### Evaluate
You can evaluate the generated samples using the script we provided in `evaluations`. We also provide our [generated samples](https://drive.google.com/drive/folders/1gx7Vx7kvGa78taePSJ_wrkiyZrYPRwNI?usp=sharing) by the model training at timestep multiplicity 32 and augmentation multiplicity 4. And the privacy budget is at ($\epsilon$, $\delta$)=(10,10<sup>-5</sup>).
