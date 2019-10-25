# Tracking by Animation: Unsupervised Learning of Multi-Object Attentive Trackers

- [CVPR 2019 version](http://openaccess.thecvf.com/content_CVPR_2019/html/He_Tracking_by_Animation_Unsupervised_Learning_of_Multi-Object_Attentive_Trackers_CVPR_2019_paper.html).
- [Full version](https://www.researchgate.net/profile/Zhen_He21/publication/332246376_Tracking_by_Animation_Unsupervised_Learning_of_Multi-Object_Attentive_Trackers/links/5ca98b864585157bd32878f1/Tracking-by-Animation-Unsupervised-Learning-of-Multi-Object-Attentive-Trackers.pdf?origin=publication_detail) (with the appendix).


**NOTES**:
- The DukeMTMC's official website was closed in 05/2019. It might be recovered in the future.


## 1. Results


### 1.1 MNIST-MOT



#### a) Qualitative results

<p align="center">
    <a href="https://youtu.be/5UNi8mhsmR4" target="_blank"><img src="imgs/mnist.gif" width="500"/></a><br/>
    &uarr;&uarr;&uarr;&emsp;Click it for a longer watch on YouTube&emsp;&uarr;&uarr;&uarr;<br/>
    Left: input. Middle: reconstruction. Right: memory (Row 1), attention (Row 2), and output (Row 3).
</p>

#### b) Quantitative results
|Configuration|IDF1&uarr;|IDP&uarr;|IDR&uarr;|MOTA&uarr;|MOTP&uarr;|FAF&darr;|MT&darr;|ML&darr;|FP&darr;|FN&darr;|IDS&darr;|Frag&darr;|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|TBA          |99.6      |99.6     |99.6     |99.5      |78.4      |0        |978     |0       |49      |49      |22       |7         |



### 1.2 Sprites-MOT


#### a) Qualitative results

<p align="center">
    <a href="https://youtu.be/hrkus5brD_U" target="_blank"><img src="imgs/sprite.gif" width="500"/></a><br/>
    &uarr;&uarr;&uarr;&emsp;Click it for a longer watch on YouTube&emsp;&uarr;&uarr;&uarr;<br/>
    Left: input. Middle: reconstruction. Right: memory (Row 1), attention (Row 2), and output (Row 3).
</p>

#### b) Quantitative results
|Configuration|IDF1&uarr;|IDP&uarr;|IDR&uarr;|MOTA&uarr;|MOTP&uarr;|FAF&darr;|MT&darr;|ML&darr;|FP&darr;|FN&darr;|IDS&darr;|Frag&darr;|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|TBA          |99.2      |99.3     |99.2     |99.2      |79.1      |0.01     |985     |1       |60      |80      |30       |22        |




### 1.3 DukeMTMC


#### a) Qualitative results

<p align="center">
    <a href="https://youtu.be/9EAXPfkuA8U" target="_blank"><img src="imgs/duke.gif" width="500"/></a><br/>
    &uarr;&uarr;&uarr;&emsp;Click it for a longer watch on YouTube&emsp;&uarr;&uarr;&uarr;<br/>
    Rows 1 and 4: input. Row 2 and 5: reconstruction. Rows 3 and 6: output.
</p>

#### b) Quantitative results
|Configuration|IDF1&uarr;|IDP&uarr;|IDR&uarr;|MOTA&uarr;|MOTP&uarr;|FAF&darr;|MT&darr;|ML&darr;|FP&darr;|FN&darr;|IDS&darr;|Frag&darr;|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|TBA      |82.4      |86.1     |79.0     |79.6      |80.4      |0.09     |1,026    |46      |64,002  |151,483 |875      |1,481     |

Quantitative results are hosted at [https://motchallenge.net/results/DukeMTMCT](https://motchallenge.net/results/DukeMTMCT), where our TBA tracker is named as ‘MOT_TBA’.


## 2. Requirements
- Python 3.7
- PyTorch 1.2/1.3/1.4
- [py-motmetrics](https://github.com/cheind/py-motmetrics) (to evaluate tracking performances)



## 3. Usage


### 3.1 Generate training data

Enter the project root directory `cd path/to/tba`.

For mnist and sprite:
```
python scripts/gen_mnist.py     # for mnist
python scripts/gen_sprite.py    # for sprite
```

For duke:
```
bash scripts/mts2jpg.sh 1                     # convert .mts files to .jpg files, please run over all cameras by setting the last argument to 1, 2, ..., 8
./scripts/build_imbs.sh                       # build imbs for background extraction
cd imbs/build
./imbs -c 1                                   # run imbs, please run over all cameras by setting c = 1, 2, ..., 8
cd ../..
python scripts/gen_duke_bb.py --c 1           # generate bounding box masks, please run over all cameras by setting c = 1, 2, ..., 8
python scripts/gen_duke_bb_bg.py --c 1        # refine background images, please run over all cameras by setting c = 1, 2, ..., 8
python scripts/gen_duke_roi.py                # generate roi masks
python scripts/gen_duke_processed.py --c 1    # resize images, please run over all cameras by setting c = 1, 2, ..., 8
python scripts/gen_duke.py                    # generate .pt files for training
```


### 3.2 Train the model


```
python run.py --task mnist     # for mnist
python run.py --task sprite    # for sprite
python run.py --task duke      # for duke
```


### 3.3 Show training curves


```
python scripts/show_curve.py --task mnist     # for mnist
python scripts/show_curve.py --task sprite    # for sprite
python scripts/show_curve.py --task duke      # for duke
```


### 3.4 Evaluate tracking performances

#### a) Generate test data
```
python scripts/gen_mnist.py --metric 1         # for mnist
python scripts/gen_sprite.py --metric 1        # for sprite
python scripts/gen_duke.py --metric 1 --c 1    # for duke, please run over all cameras by setting c = 1, 2, ..., 8
```

#### b) Generate tracking results
```
python run.py --init sp_latest.pt --metric 1 --task mnist                     # for mnist
python run.py --init sp_latest.pt --metric 1 --task sprite                    # for sprite
python run.py --init sp_latest.pt --metric 1 --task duke --subtask camera1    # for duke, please run all subtasks from camera1 to camera8
```

#### c) Convert the results into `.txt`
```
python scripts/get_metric_txt.py --task mnist                     # for mnist
python scripts/get_metric_txt.py --task sprite                    # for sprite
python scripts/get_metric_txt.py --task duke --subtask camera1    # for duke, please run all subtasks from camera1 to camera8
```

#### d) Evaluate tracking performances
```
python -m motmetrics.apps.eval_motchallenge data/mnist/pt result/mnist/tba/default/metric --solver lap      # form mnist
python -m motmetrics.apps.eval_motchallenge data/sprite/pt result/sprite/tba/default/metric --solver lap    # form sprite
```

To evaluate duke, please upload the file `duke.txt` (under `result/duke/tba/default/metric/`) to https://motchallenge.net.
