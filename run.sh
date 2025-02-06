
mkdir ./weights
cd ./weights 
wget https://huggingface.co/facebook/sapiens-pretrain-0.3b-torchscript/resolve/main/sapiens_0.3b_epoch_1600_torchscript.pt2
cd ..
# Should install requirement
python3 inference