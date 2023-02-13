# COVID-19 Chest CT Image Segmentation of Anomalies Using Fully Convolutional DenseNet with Max Unpooling

## Getting Started


### Prerequisites

* Create a folder `results` to store results and a folder `figures` to store figures

* Create a folder `dataset`, download data from [here](https://www.kaggle.com/c/covid-segmentation/data), and put them into the folder `dataset`

Your `dataset` directory should look like this:

````
-- dataset
   |-- images_medseg.npy
   |-- images_radiopedia.npy
   |-- masks_medseg.npy
   |-- masks_radiopedia.npy
   
````
* Create a folder `weights` that can store all the model weights

Your `weights` directory should look like this:

````
-- weights
   |-- FCDenseNet
   |   |-- 3classes
   |   |-- 4classes
   |-- FCDenseNetV2
   |   |-- 3classes
   |   |-- 4classes

````

* Finally, your current directory should look like this:

````
-- DenseNet_DenseNet_V2_implementation.py
-- dataset
   |-- images_medseg.npy
   |-- images_radiopedia.npy
   |-- masks_medseg.npy
   |-- masks_radiopedia.npy
   |-- test_images_medseg.npy
-- evaluation.py
-- figures
-- models
   |-- layers.py
   |-- tiramisu.py
-- plots.py
-- README.md
-- requirements.txt
-- results
-- utils
   |-- training.py
-- weights
   |-- FCDenseNet
   |   |-- 3classes
   |   |-- 4classes
   |-- FCDenseNetV2
   |   |-- 3classes
   |   |-- 4classes
````

### Running

* Install dependencies

````
pip install -r requirements.txt
````

* Train

An example to train a 3-class FCDenseNet:

````
python3 ./DenseNet_DenseNet_V2_implementation.py -t -o -w ./weights/FCDenseNet/3classes/ -l FCDenseNet_3classes -e 100  -b 4 -k 2 -d 5
````

    * t - 3-class segmentation
    * f - 4-class segmentation
    * o - FCDenseNet
    * u - FCDenseNet V2
    * d - a device for GPU computing
    * w - file path for saving weights
    * l - file name for saving figures
    * e - total number of epochs
    * b - batch size
    * k - number of dense blocks on each path, excluding bottleneck

* Test

An example to test a 3-class FCDenseNet:

````
python3 evaluation.py -t -o -w ./weights/FCDenseNet/3classes/ -k 2 -d 5
````

    * t - 3-class segmentation
    * f - 4-class segmentation
    * o - FCDenseNet
    * u - FCDenseNet V2
    * d - a device for GPU computing
    * w - file path for loading weights
    * k - number of dense blocks on each path, excluding bottleneck

* Figures and Results


An example to compare a 3-class FCDenseNet with a 3-class FCDenseNet V2:

````
python3 plots.py -l "./results/2_densetNet binary class.csv" -a "./results/2_densetNet V2 binary class.csv" -i "GGO + Consolidations" -n "2_block_binary"
````

    * l - saved results for FCDenseNet in .csv format
    * a - saved results for FCDenseNet V2 in .csv format
    * i - title for the figure
    * n - filename for the figure

## References/Citations

* [FC DenseNet](https://github.com/quannm3110/FC_DenseNet_Tiramisu)

````
@inproceedings{jegou2017one,
  title={The one hundred layers tiramisu: Fully convolutional densenets for semantic segmentation},
  author={J{\'e}gou, Simon and Drozdzal, Michal and Vazquez, David and Romero, Adriana and Bengio, Yoshua},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={11--19},
  year={2017}
}
````
