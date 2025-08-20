# Use condaforge/miniforge3 base image
FROM condaforge/miniforge3:24.11.3-2

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV CONDA_OVERRIDE_CUDA=12.0
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    # cmake \
    # ninja-build \
    # git \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment file
COPY docker/benchmark_environment.yml /tmp/benchmark_environment.yml

# Create conda environment
RUN conda env create -f /tmp/benchmark_environment.yml

# For some reason conda is silently skipping scikit-build-core and cmake installation
# Install missing dependencies that might not be installed due to conflicts
RUN conda run -n benchmark conda install -c conda-forge scikit-build-core cmake -y

# Patch pycolmap after installation
RUN sed -i 's/INVALID_POINT3D = np.uint64(-1)/INVALID_POINT3D = np.uint64(2**64-1)/' /opt/conda/envs/benchmark/lib/python3.11/site-packages/pycolmap/scene_manager.py

# Initialize conda
RUN conda init
RUN echo "conda activate benchmark" >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Clone GSplat repository
RUN git clone https://github.com/nerfstudio-project/gsplat.git gsplat

# Create directories for results
RUN mkdir -p results/benchmark/summary

# Set default conda environment
ENV CONDA_DEFAULT_ENV=benchmark
ENV PATH="/opt/conda/envs/benchmark/bin:$PATH"

# Set CUDA environment variables for compilation
ENV CUDA_HOME="/opt/conda/envs/benchmark"
ENV CUDA_PATH="/opt/conda/envs/benchmark"
ENV CPATH="/opt/conda/envs/benchmark/targets/x86_64-linux/include:$CPATH"
ENV LD_LIBRARY_PATH="/opt/conda/envs/benchmark/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
ENV LIBRARY_PATH="/opt/conda/envs/benchmark/targets/x86_64-linux/lib:$LIBRARY_PATH"

# Expose ports for visualization
EXPOSE 8080 8081
