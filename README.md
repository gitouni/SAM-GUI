#  ![](./icon.png) SAM-GUI
A GUI of [Segmentation Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) Powered by Tkinter.
* Install requirements of SAM (`pip install git+https://github.com/facebookresearch/segment-anything.git`)
* Download a SAM checkpoint from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints) and put it into `checkpoint/`
* Set relevant model path and type in [config.yml (checkpoint)](https://github.com/gitouni/SAM-GUI/blob/11ac385fd0d784f37098b93237facc5ef5dfe640/config.yml#L38)
* Put Images to be marked in `img/` (or set whatever you want in [config.yml](./config.yml))
* Run our gui (`python sam_gui.py`)
