:: Win10
:: Ananconda 3
:: How to run this script: directly input this file name into Anaconda Prompt, and Enter

set env_name=music

:: clean up
cls
call conda deactivate
call echo y|conda remove --name %env_name% --all
call del test.result

:: Tensorflow GPU 2.1.0
call echo y|conda create --name %env_name% python=3.7.6 tensorflow-gpu

:: Activate the new env
call conda activate %env_name%

:: PyTorch 1.4.0
call pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
call pip install jupyter

:: deactivate & activate the new env
call conda deactivate
call conda activate %env_name%

:: Test tensorflow-gpu
call python tf_test.py > tmp.txt
call set /p test_string=<tmp.txt
call del tmp.txt
if "%test_string%" == "True" (echo tensorflow-gpu: True > test.result) else (echo tensorflow-gpu: False > test.result)

:: Test pytorch
call python tc_test.py > tmp.txt
call set /p test_string=<tmp.txt
call del tmp.txt
if "%test_string%" == "True" (echo torch: True >> test.result) else (echo torch:  False >> test.result)

:: output test result
cls
call type test.result

:: deactivate & activate the new env
:: and install env specific packages (project specific), and add env to jupyter kernel
call conda deactivate
call conda activate %env_name%
call pip install librosa pydub ffmpeg numpy pandas seaborn sklearn
call python -m ipykernel install --user --name %env_name%