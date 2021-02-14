# Clothing Classification
This is a project that classifies the clothing items of Fashion-MNIST database using deep learning. Its aim is to achieve real time classification.

## Motivation
Creating a Know-How for deployment and usage of CNN on embedded target.

## Setup IDE
1. Install Anaconda
2. Install Visual Studio Code
3. Install python on VS Code

## Start IDE
1. Open anaconda prompt
2. Go to project file
3. On Windows: Create enviroment: conda create --name myenv; acivate myenv; FOR /F "delims=~" %f in (requirements.txt) DO conda install --yes "%f" || pip install "%f"; code .
On Mac: conda env create -n myenv -f environment.yml 
4. Run main.py

## Tests
1. To make sure that you are using the right libraries compare conda list output with libraries in environment.yml
2. When running main.py the output accuracy should be greater than 90%.

## Credits
data : https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

## ToDo:
1. turn tflite inference in C array
2. run and test inference on C/C++
3. create binary/find binary properties (size(RAM,Flash),duty cycle)
4. run bunary on chosen SoC with simulated dataset

## VS Code Commands
- ```Umschalt + ALT + F``` formatiert den Code
- ```F5``` starts debugging main.py

## License
[MIT](https://choosealicense.com/licenses/mit/)

