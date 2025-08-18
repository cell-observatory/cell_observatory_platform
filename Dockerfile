# to build the torch_cuda_12_8 image:
# docker buildx build . --tag ghcr.io/cell-observatory/cell_observatory_platform:main_torch_cuda_12_8 --build-arg BRANCH_NAME=$(git branch --show-current) --target torch_cuda_12_8 --progress=plain --no-cache-filter pip_install
#
# to run on a ubuntu system:
# install nvidia driver (distro=ubuntu2204 && arch=x86_64 && arch_ext=amd64) then follow https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu-installation and Network Repository Installation
# install docker: https://docs.docker.com/engine/install/ubuntu/
# set docker permissions for non-root: https://docs.docker.com/engine/install/linux-postinstall/ 
# install apptainer latest version from https://apptainer.org/docs/admin/main/installation.html#install-debian-packages  older version from here https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages  
# install nvidia container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# install github self-hosted runner: https://github.com/cell-observatory/cell_observatory_platform/settings/actions/runners/new?arch=x64&os=linux
# make github self-hosted runner as a service: https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/configuring-the-self-hosted-runner-application-as-a-service
# docker system prune
# container's user is different than github action's user, so change permissions of folder like this: sudo chmod 777 /home/mosaic/Desktop/actions-runner/_work -R
# install apptainer: sudo add-apt-repository -y ppa:apptainer/ppa &&  sudo apt update && sudo apt install -y apptainer

# 'conda install tensorflow-gpu' will never install GPU version because GPU is not detected during 'docker build' so we just use NVIDIA container that has it all installed already.

# this works to mount using ssh keys
# to mount clusterfs using ssh keys (1. copy keys from /.ssh on host to /sshkey in container, 2. make mount point and change permissions for local user, 3. sshfs with that key and no user input):
# docker run --rm -it --gpus all --ipc=host --cap-add=SYS_ADMIN --privileged=true --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864  -u 1000 -v ${HOME}/.ssh:/sshkey -v ${PWD}:/app/cell_observatory_platform  ghcr.io/cell-observatory/cell_observatory_platform:main_torch_cuda_12_8 /bin/bash
# sudo mkdir /clusterfs; sudo chmod a+wrx /clusterfs/; sudo chown 1000:1000 -R /sshkey/; sshfs thayeralshaabi@login.abc.berkeley.edu:/clusterfs  /clusterfs -o IdentityFile=/sshkey/id_rsa -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null

# this works to make an apptainer version
# docker run --rm kaczmarj/apptainer pull main_torch_cuda_12_8.sif docker://ghcr.io/cell-observatory/cell_observatory_platform:main_torch_cuda_12_8

# Pass in a target when building to choose the Image with the version you want: --build-arg BRANCH_NAME=$(git branch --show-current) --target torch_cuda_12_8
# For github actions, this is how we will build multiple docker images.
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-01.html#rel-25-01

# for CUDA 12.x
FROM nvcr.io/nvidia/pytorch:25.06-py3 AS base
ENV RUNNING_IN_DOCKER=TRUE

# Make bash colorful https://www.baeldung.com/linux/docker-container-colored-bash-output   https://ss64.com/nt/syntax-ansi.html 
ENV TERM=xterm-256color
RUN echo "PS1='\e[97m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc

# Install requirements. Don't "apt-get upgrade" or else all the NVIDIA tools and drivers will update.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  sudo \
  htop \
  nvitop \
  cifs-utils \
  winbind \
  smbclient \
  sshfs \
  iputils-ping \
  google-perftools \
  libgoogle-perftools-dev \
  graphviz \
  zsh \
  vmtouch \
  fio \
  prometheus \ 
  autoconf \
  libxslt-dev \ 
  xsltproc \ 
  docbook-xsl \
  && rm -rf /var/lib/apt/lists/*

RUN echo "Installing grafana"
RUN cd && \
    mkdir -p /etc/apt/keyrings/ && \
    wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | tee /etc/apt/keyrings/grafana.gpg > /dev/null  && \
    echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | tee -a /etc/apt/sources.list.d/grafana.list && \
    echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com beta main" | tee -a /etc/apt/sources.list.d/grafana.list
RUN apt-get update && \
    apt-get install -y grafana && \
    apt-get install -y grafana-enterprise && \
    rm -rf /var/lib/apt/lists/*

RUN echo "Install ohmyzsh"
RUN sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

# Give the dockerfile the name of the current git branch (passed in as a command line argument to "docker build")
ARG BRANCH_NAME

# Want to rebuild from requirements.txt everytime, so if some new dependency breaks, we catch it right away.
# Therefore we must avoid cache in this next section https://docs.docker.com/reference/cli/docker/buildx/build/#no-cache-filter
# ----- Section to be non-cached when built.
FROM base AS pip_install
COPY requirements.txt requirements.txt 
# ------

FROM pip_install AS torch_cuda_12_8
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --progress-bar off --cache-dir /root/.cache/pip

# Code to avoid running as root
ARG USERNAME=user1000
ENV USER=${USERNAME}
ARG USER_UID=1000
ARG USER_GID=1000

# Create the user
RUN groupadd --gid $USER_GID $USERNAME && \
    groupadd --gid 1001 user1000_secondary && \
    useradd -l --uid $USER_UID --gid $USER_GID -G 1001 -m $USERNAME && \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.        
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME || true

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
