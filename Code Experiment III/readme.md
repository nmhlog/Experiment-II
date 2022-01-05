# Experiment tugas 3
## list program
   ### - patching_img.py = untuk membuat patching img
   ### - training.py = untuk training
   ### - testing.py = untuk melakukan testing
   ### - Plot_Hasil.ipynb= untuk melakukan ploting hasil *manual insert

## how to:
1. download https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download/
2. export dataset : unzip warwick_qu_dataset_released_2016_07_08.zip 
3. Pastikan code data berada pada satu folder dengan keterangan seperti dibawah:
```
code 
├───training.py
├───warwick_qu_dataset_released_2016_07_08
├───patching_img.py
├───testing.py
└───Plot_Hasil.ipynb
```
4. Jalankan : python3 patching_img.py
```
├───dataset
│   ├───Test_dataset
│   │   ├───benign
│   │   └───malignant
│   └───Training_dataset
│       ├───benign
│       └───malignant
├───diff_imgs
├───Patch_dataset 
│   ├───Test_dataset
│   │   ├───benign_200x200
│   │   └───malignant_200x200
│   └───Training_dataset
│       ├───benign_200x200
│       └───malignant_200x200
└───Patch_dataset_no_ratio
    ├───Test_dataset
    │   ├───benign_200x200
    │   └───malignant_200x200
    └───Training_dataset
        ├───benign_200x200
        └───malignant_200x200
```
#### <strong>Patch_dataset merupakan folder untuk training </strong>
5. proses training :
```
python3 training -arch inception_resnet_v2  
python3 training -arch seresnet50 

argsparse lainnya:
arch='inception_resnet_v2' #pilihan untuk nama model
batch_size=16 #pilihan untuk batchsize
data='Patch_dataset/Training_dataset'#pilihan untuk dataset training
epochs=50 #pilihan untuk epoch
numlayer=16 #pilihan untuk layer freeze saat transfer learning, gunakan 0 bila tidak ingin melakaukan freze layer
pretrained=False # menggunakan pretrain
resume='Training' #folder untuk check point dan best model
```
Note program akan menampilkan folder :
```
Training/<model name>
```
yang berisikan check point dan bestmodel

6. Testing :
```
python3 testing.py -arch inception_resnet_v2  
python3 testing.py -arch seresnet50 
dengan args parse sama dengan training
```
Note folder akan menghasilkan file:
```Training/<model name>/hasilprediksi.pth``` yang berisikan hasil testing.
## requirement :
```
timm
pytorch
pandas
opencv
sklearn
numpy
matplotlib
```

other resource :
[timm github](https://github.com/rwightman/pytorch-image-models)
[timm documents](https://rwightman.github.io/pytorch-image-models/)

