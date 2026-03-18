# For amd64 support only:
#     docker buildx build . --tag ghcr.io/cell-observatory/cell_observatory_platform:main_torch_26_01--build-arg BRANCH_NAME=$(git branch --show-current) --target torch_26_01--progress=plain --no-cache-filter pip_install

# For multi-platform support:
#     1. Check if containerd-snapshotter is enabled
#         docker info -f '{{ .DriverStatus }}'
#         You should see snapshotter in the output: [[driver-type io.containerd.snapshotter.v1]]
#         If not, make sure your /etc/docker/daemon.json has containerd-snapshotter enabled https://docs.docker.com/engine/storage/containerd/

#     2. Use the tonistiigi/binfmt image to install QEMU and register the executable types on the host
#         docker run --privileged --rm tonistiigi/binfmt --install all

#     3. Build image with --platform flag:
#         docker buildx build --platform linux/amd64,linux/arm64 . --tag ghcr.io/cell-observatory/cell_observatory_platform:main_torch_26_01--build-arg BRANCH_NAME=$(git branch --show-current) --target torch_26_01--progress=plain --no-cache-filter pip_install


# Running image:
#     docker run --network host -u 1000 --privileged -v ~/.ssh:/sshkey -v ${PWD}:/workspace/cell_observatory_platform --env PYTHONUNBUFFERED=1 --pull missing -t -i --rm -w /workspace/cell_observatory_platform --ipc host --gpus all ghcr.io/cell-observatory/cell_observatory_platform:develop_torch_26_01 bash


# to run on a ubuntu system:
# install nvidia driver (distro=ubuntu2404 && arch=x86_64 && arch_ext=amd64) then follow https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu-installation and Network Repository Installation
# install docker: https://docs.docker.com/engine/install/ubuntu/
# set docker permissions for non-root: https://docs.docker.com/engine/install/linux-postinstall/ 
# install apptainer latest version from https://apptainer.org/docs/admin/main/installation.html#install-debian-packages  older version from here https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages  
# install nvidia container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# install github self-hosted runner: https://github.com/cell-observatory/cell_observatory_platform/settings/actions/runners/new?arch=x64&os=linux
# make github self-hosted runner as a service: https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/configuring-the-self-hosted-runner-application-as-a-service
# docker system prune
# container's user is different than github action's user, so change permissions of folder like this: sudo chmod 777 /home/mosaic/Desktop/actions-runner/_work -R
# install apptainer: sudo add-apt-repository -y ppa:apptainer/ppa &&  sudo apt update && sudo apt install -y apptainer

# this works to make an apptainer version
# docker run --rm kaczmarj/apptainer pull main_torch_26_01.sif docker://ghcr.io/cell-observatory/cell_observatory_platform:main_torch_26_01

# Pass in a target when building to choose the Image with the version you want: --build-arg BRANCH_NAME=$(git branch --show-current) --target torch_26_01
# For github actions, this is how we will build multiple docker images.
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html


FROM nvcr.io/nvidia/pytorch:26.01-py3 AS base
ENV RUNNING_IN_DOCKER=TRUE

# Make bash colorful https://www.baeldung.com/linux/docker-container-colored-bash-output   https://ss64.com/nt/syntax-ansi.html 
ENV TERM=xterm-256color
RUN echo "PS1='\e[97m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc


# Install requirements. Don't "apt-get upgrade" or else all the NVIDIA tools and drivers will update.
RUN apt-get update && apt-get install -y --no-install-recommends \
  sudo \
  htop \
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
  libnuma-dev \
  software-properties-common \
  postgresql-client \
  rsync \
  && rm -rf /var/lib/apt/lists/*


RUN echo "Install apptainer"
RUN add-apt-repository -y ppa:apptainer/ppa
RUN apt-get update && apt-get install -y apptainer && rm -rf /var/lib/apt/lists/*

RUN echo "Installing grafana"

RUN mkdir -p /etc/apt/keyrings/ && \
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

RUN echo "Install additional dependencies"
FROM pip_install AS torch_26_01
RUN pip install --ignore-installed cryptography
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --progress-bar off --root-user-action=ignore --cache-dir /root/.cache/pip

RUN echo "Install ops3d kernels"
RUN BUILD_ARCH=$(uname -m) && \
    if [ "$BUILD_ARCH" = "x86_64" ]; then \
        pip install https://raw.githubusercontent.com/cell-observatory/ops3d/main/dist/ops3d-0.1.0-cp312-cp312-linux_x86_64.whl; \
    elif [ "$BUILD_ARCH" = "aarch64" ]; then \
        pip install https://raw.githubusercontent.com/cell-observatory/ops3d/main/dist/ops3d-0.1.0-cp312-cp312-linux_aarch64.whl; \
    else \
        echo "Unsupported architecture"; exit 1; \
    fi
    
# Code to avoid running as root
ARG USERNAME=user1000
ENV USER=${USERNAME}
ARG USER_UID=1000
ARG USER_GID=1000

# Create the user
RUN groupadd --gid $USER_GID $USERNAME && \
    groupadd --gid 1001 user1000_secondary && \
    useradd -l --uid $USER_UID --gid $USER_GID -G 1001 -m $USERNAME && \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.        
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME || true

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]