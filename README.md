## Setup / Installation
Please use the provided conda_environment.yaml file to set up a conda environment. 

If you don't use conda, code depends on OpenAI gym, stable_baselines3, pandas, tqdm, and pytorch. Install however you see fit.


## To run pretrained models:

The classes are currently set up to run the pretrained models saved in the `output/` folder. 

Edit the `utils.py` file to pick which ENV you want to run. Comment out any other ENV variable definitions.

Run the file that corresponds to whichever model you wish to run (`dqn.py`, `m_dqn.py`, etc). 



## To train your own model:
Uncomment the lines that look like 

```python
# model.learn(...)
# model.save(...)
```
This corresponds to: 
* line 56-57 in `dqn.py`
* line 66-67 in `softdqn.py`
* line 84-85 in `m_dqn.py`
* line 80-81 in `a2c.py`
* line 79-80 in `m_a2c.py`

in the file corresponding to whichever model you wish to run (`dqn.py`, `m_dqn.py`, etc). 

Be aware this will overwrite the pretrained model. Use git to restore pretrained models as you need.
Run the file that corresponds to whichever model you wish to run (`dqn.py`, `m_dqn.py`, etc). 
