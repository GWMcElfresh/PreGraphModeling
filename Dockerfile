#bioconductor base image with R 4.51 and Bioconductor 3.21
FROM bioconductor/bioconductor_docker:RELEASE_3_21

#metadata
LABEL org.opencontainers.image.title="PreGraphModeling"
LABEL org.opencontainers.image.description="Tools for subsetting Seurat objects and fitting zero-inflated negative binomial models"
LABEL org.opencontainers.image.authors="GW McElfresh <mcelfreshgw@gmail.com>"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL org.opencontainers.image.source="https://github.com/GWMcElfresh/PreGraphModeling"

RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libgit2-dev \
    libhdf5-dev \
    libgsl-dev \
    libglpk-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#bioconductor installs
RUN R -e "BiocManager::install(c('DESeq2', 'HDF5Array', 'DelayedArray'), ask = FALSE, update = FALSE)"

#cran installs
RUN R -e "install.packages(c('SeuratObject', 'pscl', 'Matrix', 'methods', 'parallel', 'mgcv', 'stats', 'testthat', 'Seurat', 'future', 'future.apply', 'remotes', 'devtools'), repos='https://cloud.r-project.org/', dependencies=TRUE)"

#install PreGraphModeling
WORKDIR /workspace
COPY DESCRIPTION NAMESPACE LICENSE README.md ./
COPY R/ ./R/
COPY man/ ./man/
COPY tests/ ./tests/
RUN R -e "remotes::install_local('.', dependencies = FALSE, upgrade = 'never')"

#default command
CMD ["R"]
