### Environment Installation ###
1) Install python 3.5.4 using the provided .exe file in [wheels](./wheels) folder. Then, open a command line, navigate to the python 3.5.4 installation folder.
2) Create a venv with the freshly installed python 3.5.4 version: 
```
python -m venv path_where_venv_should_be_stored
```
   
3) Activate the venv by executing ```activate``` in the bash after navigating to the Script folder in the folder where the venv was installed to.
4) Install correct pip, setup and wheel package with the following commands using the wheels in the [wheels](./wheels) folder (replace path_to_wheel_folder with the correct path before executing, keep the ""):
```
python -m pip install --no-cache-dir --no-index --find-links="path_to_wheel_folder" pip==20.3.4
python -m pip install --no-cache-dir --no-index --find-links="path_to_wheel_folder" setuptools==50.3.2 wheel==0.36.2
```
5) Install [requirements_local2_withouttorch.txt](./requirements_local2_withouttorch.txt)
```
pip install -r requirements_local2_withouttorch.txt
```
6) Install torchvision using the wheel in the [wheels](./wheels) folder. Thus, execute this commands after navigating to the [wheels](./wheels) folder:
```
pip install torchvision-0.6.0+cpu-cp35-cp35m-win_amd64.whl
```
7) Install the torch wheel from [pytorch](https://download.pytorch.org/whl/cpu) using this [link](https://download.pytorch.org/whl/cpu/torch-1.5.0%2Bcpu-cp35-cp35m-win_amd64.whl). Then execute the following command in the folder containing the downloaded wheel.
```
pip install torch-1.5.0+cpu-cp35-cp35m-win_amd64.whl
```


