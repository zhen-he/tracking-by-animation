# Project Page for CVPR 2019 Submission 
# Paper ID: 648


Paper Title: **Tracking by Animation: Unsupervised Learning of Multi-Object Attentive Trackers**


<!--
The latest version of our paper is available at [**HERE**](https://github.com/anonymous-projects/tracking-by-animation).
-->


## 1. Results


### 1.1 MNIST-MOT



#### a) Qualitative results

<p align="center">
    <a href="https://vimeo.com/295500734" target="_blank"><img src="imgs/mnist.gif" width="500"/></a><br/>
    Click it to watch longer &uarr;<br/>
    Left: input. Middle: reconstruction. Right: memory (Row 1), attention (Row 2), and output (Row 3).
</p>

#### b) Quantitative results
|Configuration|IDF1&uarr;|IDP&uarr;|IDR&uarr;|MOTA&uarr;|MOTP&uarr;|FAF&darr;|MT&darr;|ML&darr;|FP&darr;|FN&darr;|IDS&darr;|Frag&darr;|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|TBA          |99.6      |99.6     |99.6     |99.5      |78.4      |0        |978     |0       |49      |49      |22       |7         |



### 1.2 Sprites-MOT


#### a) Qualitative results

<p align="center">
    <a href="https://vimeo.com/295500903" target="_blank"><img src="imgs/sprite.gif" width="500"/></a><br/>
    Click it to watch longer &uarr;<br/>
    Left: input. Middle: reconstruction. Right: memory (Row 1), attention (Row 2), and output (Row 3).
</p>

#### b) Quantitative results
|Configuration|IDF1&uarr;|IDP&uarr;|IDR&uarr;|MOTA&uarr;|MOTP&uarr;|FAF&darr;|MT&darr;|ML&darr;|FP&darr;|FN&darr;|IDS&darr;|Frag&darr;|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|TBA          |99.2      |99.3     |99.2     |99.2      |79.1      |0.01     |985     |1       |60      |80      |30       |22        |




### 1.3 DukeMTMC


#### a) Qualitative results

<p align="center">
    <a href="https://vimeo.com/295501114" target="_blank"><img src="imgs/duke.gif" width="500"/></a><br/>
    Click it to watch longer &uarr;<br/>
    Rows 1 and 4: input. Row 2 and 5: reconstruction. Rows 3 and 6: output.
</p>

#### b) Quantitative results
|Configuration|IDF1&uarr;|IDP&uarr;|IDR&uarr;|MOTA&uarr;|MOTP&uarr;|FAF&darr;|MT&darr;|ML&darr;|FP&darr;|FN&darr;|IDS&darr;|Frag&darr;|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|TBA          |82.0      |85.0     |79.2     |78.0      |80.0      |0.11     |1018     |40      |79,402  |152,336 |803      |1,466     |

Quantitative results are made public at [https://motchallenge.net/results/DukeMTMCT](https://motchallenge.net/results/DukeMTMCT) ('MOT\_TBA' with paper ID 648).


## 2. Requirements
- python 3.6
- pytorch 0.3.1
- [py-motmetrics](https://github.com/cheind/py-motmetrics) (to evaluate tracking performances)



## 3. Usage


### 3.1 Generate training data

```
cd path/to/tba                  # enter the project root directory
python scripts/gen_mnist.py     # for mnist
python scripts/gen_sprite.py    # for sprite
python scripts/gen_duke.py      # for duke
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
