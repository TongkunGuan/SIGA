# Self-Supervised Implicit Glyph Attention for Text Recognition (CVPR2023)
This is the official code of "Self-Supervised Implicit Glyph Attention for Text Recognition". 
For more details, please refer to our [CVPR2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guan_Self-Supervised_Implicit_Glyph_Attention_for_Text_Recognition_CVPR_2023_paper.pdf) 
or [Poster](SIGA_poster.pdf) or [中文解读](https://www.techbeat.net/article-info?id=5152). If you have any questions please contact me by email (gtk0615@sjtu.edu.cn).

We also released ICCV23 work on scene text recognition:
- Self-supervised Character-to-Character Distillation for Text Recognition（CCD）
[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Guan_Self-Supervised_Character-to-Character_Distillation_for_Text_Recognition_ICCV_2023_paper.pdf) and [Code](https://github.com/TongkunGuan/CCD)


## Pipeline 
![examples](graph/Essentialdifference.png)

## Model architecture
![examples](graph/Network.png)

## Environments
```bash
# V100 Ubuntu 16.04 Cuda 10
conda create -n SIGA python==3.7.0
source activate SIGA
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard==1.15.0
pip install tensorboardX==2.2
pip install opencv-python
pip install Pillow LMDB nltk six natsort scipy
# 3090 Ubuntu 16.04 Cuda 11
conda create -n SIGA python==3.7.0
source activate SIGA
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install tensorboard==2.11.2
pip install tensorboardX==2.2
pip install opencv-python
pip install Pillow LMDB nltk six natsort scipy
# if you meet bug about setuptools
# pip uninstall setuptools
# pip install setuptools==58.0.4
```
## Data
```bash
-- root_path
    --training
        --MJ
        --ST
    --validation
    --evaluation
        --SVT
        --IIIK
        --...
```

## Highlights
- **Dataset link:**
  - [Synth data and benchmark](https://github.com/FangShancheng/ABINet)
  - [MPSC](https://drive.google.com/file/d/1awBUrj30s5VLxk-B0EFkieeIcNrETVTX/view?usp=drive_link)
    ![examples](graph/MPSC.png)
  - [ArbitText](https://drive.google.com/file/d/1D-pN5u0KOm79Ot51jh-9_rEdSt9oK_vK/view?usp=drive_link)
    ![examples](graph/ArbitText.png)
- **weight link:**
  - [SIGA<sub>R</sub>](https://drive.google.com/file/d/1Nulplr3LnMzJ1AgG5pkCyUCUeqddR0_P/view?usp=drive_link) *The model is trained on V100 platform*.
  - [SIGA<sub>S</sub>](https://drive.google.com/file/d/1o1trg0yQWVmycEuFqa4IsKLOrPJP1TbN/view?usp=drive_link) *The model is trained on 3090 platform*.
  - [SIGA<sub>T</sub>](https://drive.google.com/file/d/14wCXJg_dLFxDuEnVBpJH5MRzLh_U9meK/view?usp=drive_link) *The model is trained on 3090 platform*.

## Mask preparation
  - optional, K-means results (please refer to [CCD](https://github.com/TongkunGuan/CCD/tree/main/mask_create))
```bash
cd ./mask_create
run generate_mask.py #parallelly process mask --> lmdb file
run merge.py #merge multiple lmdb files into single file
```
  
## Training 
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --model_name TRBA --exp_name SIGA --Aug --batch_size 320 --num_iter 160000 --select_data synth --benchmark_all_eval --train_data /xxx/dataset/data_lmdb/training/label/Synth/ --eval_data /xxx/dataset/data_lmdb/evaluation/ --mask_path /xxx/dataset/data_lmdb/Mask(optional) --workers 12
```

## Test and eval
```bash
python test.py --eval_data /xxx/xxx --select_data xxx
```

### TODO
- [ ] Refactor and clean code

## Citation
```bash
If you find our method useful for your reserach, please cite

@inproceedings{guan2023self,
  title={Self-Supervised Implicit Glyph Attention for Text Recognition},
  author={Guan, Tongkun and Gu, Chaochen and Tu, Jingzheng and Yang, Xue and Feng, Qi and Zhao, Yudi and Shen, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15285--15294},
  year={2023}
}
```
## License
```bash
- This code are only free for academic research purposes and licensed under the 2-clause BSD License - see the LICENSE file for details.
```
