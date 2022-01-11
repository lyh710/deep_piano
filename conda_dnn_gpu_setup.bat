:: Win10
:: Ananconda 3
:: How to run this script: directly input this file name into Anaconda Prompt, and Enter

set env_name=piano

:: clean up
cls
call conda deactivate
call echo y|conda remove --name %env_name% --all

:: create new conda env
call echo y|conda create --name %env_name% python=3.9

:: deactivate & activate the new env
call conda deactivate
call conda activate %env_name%
call echo y|conda install cudnn
call pip install tensorflow

:: deactivate & activate the new env
:: and install env specific packages (project specific), and add env to jupyter kernel
call conda deactivate
call conda activate %env_name%
call pip install numpy pandas seaborn sklearn progressbar2 music21 beautifulsoup4 nbstripout
call nbstripout --install
call python -m ipykernel install --user --name %env_name%