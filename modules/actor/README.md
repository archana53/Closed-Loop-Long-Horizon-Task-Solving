# Low Level Action Execution after LLMs
## How to use it:
1. Setup virtualenv and install requirements:
```bash
# setup virtualenv with whichever package manager you prefer
virtualenv -p $(which python3.8) --system-site-packages cliport_env  
source cliport_env/bin/activate
pip install --upgrade pip

pip install -r requirements.txt

export CLIPORT_ROOT=$(pwd)
python setup.py develop
```
2. Download a [pre-trained checkpoint](https://drive.google.com/file/d/1w8yzqrIf-bTXp6NazQ_o8V-xiJB3tlli/view?usp=sharing) for `multi-language-conditioned` trained with 1000 demos:
```bash
python quickstart_download.py
``` 
3. Open *affordances.ipynb*
4. Change the *lang_goals* value with LLM results
5. Check the affordance results

## TODO:
1. Only works for Task "stack-block-pyramid-seq-seen-colors", need to modify the code for all tasks
2. Clean up the code and write a new notebook to go through the whole running process

## Reference:
[Cliport](https://github.com/cliport/cliport)
