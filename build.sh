cd software/mmdet/;
python setup.py develop;
cd ../mmdet3d/;
python setup.py develop;
cd ../mmseg/;
python setup.py develop;
cd ../../det3d/models/backbones/DCNv2_t18/;
bash make.sh

