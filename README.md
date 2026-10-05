# SC-ImageNet
This is a publicly available dataset associated with our paper 'Weakly Supervised Learning of Semantic Correspondence through Cascaded Online Correspondence Refinement'. 

## Agreement
- The SC-ImageNet dataset is available solely for **non-commercial research purposes**.
- All images in the SC-ImageNet dataset are sourced from the ImageNet dataset and are not the property of the School of Computer Science, Fudan University. Our group is neither responsible for the content nor the meaning of these images.
- You agree **not to** reproduce, duplicate, copy, sell, trade, resell, or exploit any part of the images or any derivative data, including but not limited to annotations and cropped image parts, for any commercial purposes.  
- You agree **not to** further copy, publish, or distribute any portion of the SC-ImageNet dataset. However, for internal use at a single site within the same organization, it is permissible to make copies of our dataset.
- Our group reserves the right to terminate your access to the SC-ImageNet dataset at any time.

## Link
Please send an email to yiwenhuang21@m.fudan.edu.cn with a **signed agreement** before gaining access to this dataset.

- Baidu Netdisk: https://pan.baidu.com/s/1DRCEeKDUseEtXbV7jJJs4A (key: jpxc)
- Google Cloud: https://drive.google.com/file/d/1J0iZqcKWDttoJlpc0DwJH9NbX5L3dS8h/view?usp=sharing

## Data Organization

Extract the SC-ImageNet annotations and split lists into `SC-IMAGENET`.

Download the matching ImageNet images separately and organize the data as follows:

```text
<datapath>/SC-IMAGENET/
    Layout/large/trn.txt
    Layout/large/val.txt
    PairAnnotation/trn/<pair_id>.json
    PairAnnotation/val/<pair_id>.json
    JPEGImages/<category>/<category>_<image_id>.JPEG
```

On Linux, create a symbolic link so both names point to the same dataset:

```bash
ln -s SC-IMAGENET "<datapath>/SC-ImageNet"
```

Replace `<datapath>` with your dataset parent directory. The `SC-ImageNet` link path must not already exist.

If you already have the full ImageNet images extracted into category directories with the filenames shown above, link that image directory to `JPEGImages` to reuse the images without copying them:

```bash
ln -s "/absolute/path/to/imagenet/images" "<datapath>/SC-IMAGENET/JPEGImages"
```

Replace `/absolute/path/to/imagenet/images` with the absolute path to the directory containing the category folders, such as `n02138441/`. The `JPEGImages` link path must not already exist. Keep `Layout` and `PairAnnotation` from SC-ImageNet in `SC-IMAGENET`.

## Repository Setup

Copy the files from the corresponding directory into the root of the original repository, preserving the directory structure and replacing the matching files where necessary.

Then, follow the installation and setup instructions provided by the original project.

To use SC-ImageNet, select `--benchmark imagenet` and set `--datapath` to the parent directory containing `SC-IMAGENET`. 

## Model Weights

The SC-ImageNet column provides weights pretrained on SC-ImageNet. 

The SPair-71k column provides weights pretrained on SC-ImageNet and then fine-tuned on SPair-71k.

| Model | SC-ImageNet | SPair-71k |
| --- | --- | --- |
| CATs | [Google Drive](https://drive.google.com/drive/folders/1ANCg93djd3cyRye64jzrUDqjm3vLQrku?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/1QO5YMuA9KdfNFeREEhvTwaNXtb9nW67_?usp=sharing) |
| DHPF | [Google Drive](https://drive.google.com/drive/folders/1zF3ZiUgGpiFjCEohJ8wpX1HBK6VVGjWE?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/1cWDAuUAnCGPCbwoyCTKSFBGjs0lhWrCx?usp=sharing) |

## Cite
```
@InProceedings{Huang_2023_ICCV,
    author    = {Huang, Yiwen and Sun, Yixuan and Lai, Chenghang and Xu, Qing and Wang, Xiaomei and Shen, Xuli and Ge, Weifeng},
    title     = {Weakly Supervised Learning of Semantic Correspondence through Cascaded Online Correspondence Refinement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16254-16263}
}
```
