# Bayesian Optimization with Adaptive Level-Set Estimation (BALLET)
Project for Protein study. README written in Markdown. The PDF file for the recent submission to ICML ReALML workshop is attached in the [folder](./zhang22ballet.pdf).

## Environment
The implementation has been verfied on **python 3.9.7 MacOS 12.4 (M1 Pro)**.
The required packages are included in requirements.txt, to install all required packages, please run the following code.

```shell
pip install -r requirements.txt
```

## Guidance
One example is shown below. The algorithm will extract the data from the directory specified by datadir and return the suggested candidate to evaluate.

```shell
python main.py --name="exp_name" --aedir="./tmp/tmp_ae" --subdir="./res/"  --datadir="./data/Capacity_example.xlsx" --batch-size=5 --train_times=10 --beta=10 --acq_func="ucb" --learning_rate=6 -f -a;

# example of corresponding output:
       Biomass source  Highest temp. (oC)  Fe load (mC%)  KOH load (mC%)
0                MxG               800.0            6.0            60.0
1  Commercial Lignin              1100.0            1.0            40.0
2               Hemp              1000.0           12.0           100.0
3               Hemp              1200.0            6.0            40.0
4  Commercial Lignin              1100.0            3.0            40.0
```


***Note***: valid value of each attributes (2160 in toal)
```
_biomass_categories     = ['Switchgrass', 'Hemp', 'MxG', 'Commercial Lignin']
_temp                   = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
_fe                     = [200, 100, 50, 25, 12, 6, 3, 2, 1]
_koh                    = [0, 20, 40, 60, 80, 100]
```

