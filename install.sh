sudo apt-get install python3.7 python3.7-dev
sudo apt install python3-pip
pip3 install pandas --user --no-cache-dir
pip3 install -U nltk
pip3 install -U scikit-learn
pip3 install numpy --user --no-cache-dir
pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html --user --no-cache-dir

# pip3 install setuptools --user --no-cache-dir
# pip3 install wheel --user --no-cache-dir
# pip3 install ninja --user --no-cache-dir
# pip3 install joblib --user --no-cache-dir
# pip3 install torch --user --no-cache-dir
# pip3 install torchvision --user --no-cache-dir
# pip3 install torchwordemb --user --no-cache-dir
# pip3 install torchtext --user --no-cache-dir
# pip3 install matplotlib --user --no-cache-dir

pip3 install tensorflow --user --no-cache-dir
pip3 install spacy --user --no-cache-dir
python3 -m spacy download en_core_web_lg

pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz --user --no-cache-dir
