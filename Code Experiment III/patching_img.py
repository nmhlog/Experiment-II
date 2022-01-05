import pandas as pd
import cv2 as cv
from glob import glob
import numpy as np
import os 
import shutil

def get_different_imgs(output_folder,folder1='Patch_dataset_v2\\Test_dataset\\malignant200x200',folder2='Dataset_patch\\Test\\malignant_200x200'):
    """fungsi untuk membandingkan hasil patch algortihm

    Args:
        output_folder ([dir]): folder output penyimpanan files
        folder1 ([dir], optional): folder basedline gambar yang dicopy untuk mendapatkan perbandingan dari folder2. 
                                    Defaults to 'Patch_dataset_v2\Test_dataset\malignant200x200'.
        folder2 ([dir], optional): folder pembanding. Defaults to 'Dataset_patch\Test\malignant_200x200'.
    """
    folder1_list = [i.split('\\')[-1] for i in glob(f"{folder1}\\**")]
    folder2_list = [i.split('\\')[-1] for i in glob(f"{folder2}\\**")]

    data_diff = []
    
    for i in folder1_list:
        if i not in folder2_list:
            data_diff.append(i)
    os.makedirs(output_folder,exist_ok=True)

    for i in data_diff:
        shutil.copy2(f'{folder1}\\'+i,f'{output_folder}\\'+i)
        
def split_data_basedonlabel(pandas_data,dict_label,output_dir,input_dir="Warwick QU Dataset (Released 2016_07_08)"):
    for i in dict_label.keys():
        buff_dir = output_dir+"\\"+i.strip()+"\\"
        os.makedirs(buff_dir,exist_ok=True)
        for data in list(pandas_data[pandas_data["Label"] == i]["Img_name"]):
            shutil.copy("Warwick QU Dataset (Released 2016_07_08)"+f"\\{data}.bmp",buff_dir+f"{data}.bmp")
            
def sliding_window(image, stepSize):
    """
    sliding windows algorithm 

    Args:
        image (array_of image): gambar dalam bentuk array
        stepSize (int): ukuran stepsize

    Yields:
        return index of sliding windows
    """
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y)
      
def get_patch_img(datas_input_path,
                  datas_output_path,
                  size,
                  ratio_white_color,
                  step_size,
                  used_gray_channel =True):
    """[summary]

    Args:
        datas_input_path ([dir]): string input dataset
        datas_output_path ([dir]): string output dataset
        size ([int]): output ukuran gambar
        ratio_white_color ([float]): ratio warna putih [0:1], dengan memasukan nilai 1 
                                    berarti tidak menggunakan rasio warna putih
        step_size ([int]): ukuran stepsize
        used_gray_channel (bool, optional): pemilihan channel warna. Defaults to True.
    """
    patch_size = size
    list_data= glob(f"{datas_input_path}\\**.bmp")
    # os.makedirs(datas_output_path,exist_ok=True)
    for data in list_data:
        print(data)
        name = data.split("\\")[-1].split(".")[0]
        img = cv.imread(data,cv.COLOR_BGR2RGB)
        windows = sliding_window(img, step_size)
        for index in windows:
            # print("index")
            x,y= index
            cropped = img[x:x+patch_size,y:y+patch_size,:]

            if cropped.shape ==(patch_size,patch_size,3) :
                if used_gray_channel:
                    img_gray= cv.cvtColor(cropped,cv.COLOR_BGR2GRAY)
                    percentage_white_img = img_gray.flatten()
                else :
                    percentage_white_img = cropped.flatten()
                try:
                    percentage_white_img =  len(percentage_white_img[percentage_white_img>=245])/len(percentage_white_img)
                except:
                    percentage_white_img = 0
                if percentage_white_img<ratio_white_color: 
                    cv.imwrite(f"{datas_output_path}\\{name}_{x}_{y}_.bmp", cropped)


def start_patching_dataset(root_folder,
                           step_size=30,
                           save_folder="Patch_dataset",
                           patch_size =200,
                           ratio_white_color=0.7,
                           used_gray_channel =True):
    """
    algoritma untuk memulai generated dataset
    algoritma berkerja untuk folder root dataset yang telah membagi img berdasarkan folder label

    Args:
        root_folder ([dir]): folder root dataset
        step_size (int, optional): ukuran stepsize. Defaults to 30.
        save_folder ([dir], optional): folder penyimpanan output. Defaults to "Patch_dataset".
        patch_size (int, optional): ukuran output dataset. Defaults to 200.
        ratio_white_color (float, optional): ratio warna putih [0:1], dengan memasukan nilai 1 
                                    berarti tidak menggunakan rasio warna putih. Defaults to 0.7.
        used_gray_channel (bool, optional): pemilihan channel warna. Defaults to True.
    """
    for dirs in os.listdir(root_folder):
        dirs_path = os.path.join(root_folder,dirs)
        for dir in os.listdir(dirs_path):
            path = os.path.join(root_folder,dirs,dir)
            path_output = os.path.join(save_folder,dirs,dir+f"_{patch_size}x{patch_size}")
            os.makedirs(path_output,exist_ok=True)
            print("On Process data",end="  |  ")
            print(dir)
            get_patch_img(datas_input_path=path,
                          datas_output_path=path_output,
                          size=patch_size,
                          ratio_white_color=ratio_white_color,
                          step_size=step_size,
                          used_gray_channel=used_gray_channel)
            
if __name__ =="__main__":
    data_set = "Warwick QU Dataset (Released 2016_07_08)"
    pd_grade = f"{data_set}/Grade.csv"
    pd_grade= pd.read_csv(pd_grade)
    pd_grade["category"] = [ i.split("_")[0] for i in pd_grade["name"]] 
    pd_grade= pd_grade[[pd_grade.columns[-1],
                    pd_grade.columns[2],
                    pd_grade.columns[0]]]
    pd_grade.rename({pd_grade.columns[0]:"Category",pd_grade.columns[1]:"Label",pd_grade.columns[2]:"Img_name"}, axis=1, inplace=True)
    training_data = pd_grade[pd_grade["Category"]=="train"]
    test_data = pd_grade[pd_grade["Category"]!="train"]
    dict_label = {val :i for i,val in enumerate(pd.unique(training_data["Label"]))}
    split_data_basedonlabel(training_data,dict_label,'dataset/Training_dataset')
    split_data_basedonlabel(test_data,dict_label,'dataset/Test_dataset')
    start_patching_dataset('dataset',save_folder="Patch_dataset",used_gray_channel=False,ratio_white_color=0.1)
    start_patching_dataset('dataset',save_folder="Patch_dataset_no_ratio",used_gray_channel=False,ratio_white_color=1)
    get_different_imgs(output_folder="diff_imgs",
                   folder1='Patch_dataset_no_ratio\\Test_dataset\\malignant_200x200',
                   folder2='Patch_dataset\\Test_dataset\\malignant_200x200')