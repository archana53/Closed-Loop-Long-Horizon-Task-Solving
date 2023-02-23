# Record of how the environment was set up
# Create conda environment. Mamba is recommended for faster installation.
conda_env_name=dlm_project
mamba create -n $conda_env_name python=3.7 cmake=3.14.0 -y

# Install this repo as a package
mamba activate $conda_env_name

#LLM dependencies
pip install ftfy regex tqdm fvcore imageio==2.4.1 imageio-ffmpeg==0.4.5
pip install git+https://github.com/openai/CLIP.git
pip install -U --no-cache-dir gdown --pre
pip install pybullet moviepy
pip install flax==0.5.3
pip install openai
pip install easydict
pip install imageio-ffmpeg
gsutil cp -r gs://cloud-tpu-checkpoints/detection/projects/vild/colab/image_path_v2 ./weights