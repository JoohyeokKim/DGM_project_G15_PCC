Our baseline is from SP-GAN: https://github.com/liruihui/sp-gan
Most of our code is based on it.

Chamfer3D is from https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git
Since we do not need other dimension's loss, we just brought Chamfer 3D.
To use this library, move to "./metrics/Chamfer3D" and type
$ python setup.py install

Conda environment is in environments.yaml file.

To prepare dataset, you can download them from https://drive.google.com/file/d/1q8UqOBS00hkhP_cdjHqT3GNwzN5P1qZy/view?usp=sharing
Please download it and unzip on the main folder.

To run training, please type

$ bash training_prompt.sh

To run test, please type
$ bash testdgm.py
