# R PreGraphModeling Dockerfile with Bioconductor
# Multi-stage build: base -> deps -> runtime
# Compatible with GWMcElfresh/dockerDependencies caching workflows

ARG DEPS_IMAGE=deps
ARG BASE_IMAGE=bioconductor/bioconductor_docker:RELEASE_3_21

# ============================================================================
# Stage 1: Base - System dependencies
# ============================================================================
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG SKIP_BASE_DEPS=false

# Metadata
LABEL org.opencontainers.image.title="PreGraphModeling"
LABEL org.opencontainers.image.description="Tools for subsetting Seurat objects and fitting zero-inflated negative binomial models"
LABEL org.opencontainers.image.authors="GW McElfresh <mcelfreshgw@gmail.com>"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL org.opencontainers.image.source="https://github.com/GWMcElfresh/PreGraphModeling"

# Install system dependencies (skipped if using pre-built base image)
RUN if [ "$SKIP_BASE_DEPS" = "false" ]; then \
    apt-get update && apt-get install -y \
        libxml2-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libgit2-dev \
        libhdf5-dev \
        libgsl-dev \
        libglpk-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Skipping base dependency installation (using pre-built base image)"; \
    fi

# ============================================================================
# Stage 2: Deps - R packages
# ============================================================================
FROM base AS deps

ARG DEBIAN_FRONTEND=noninteractive
ARG GH_PAT='NOT_SET'

# Bioconductor installs
RUN R -e "BiocManager::install(c('DESeq2', 'HDF5Array', 'DelayedArray'), ask = FALSE, update = FALSE)"

# CRAN installs
RUN R -e "install.packages(c('SeuratObject', 'pscl', 'Matrix', 'methods', 'parallel', 'mgcv', 'stats', 'testthat', 'Seurat', 'future', 'future.apply', 'remotes', 'devtools'), repos='https://cloud.r-project.org/', dependencies=TRUE)" && \
    rm -rf /tmp/downloaded_packages/ /tmp/*.rds

# ============================================================================
# Stage 3: Runtime - Application code
# ============================================================================
FROM ${DEPS_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Copy R package files
COPY DESCRIPTION NAMESPACE LICENSE README.md ./
COPY R/ ./R/
COPY man/ ./man/
COPY tests/ ./tests/

# Build and install PreGraphModeling
RUN R CMD build . && \
    R CMD INSTALL --build *.tar.gz && \
    rm -rf /tmp/downloaded_packages/ /tmp/*.rds

CMD ["R"]
