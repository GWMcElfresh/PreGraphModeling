# Multi-stage Dockerfile for PreGraphModeling
# Stage 1: Base - System dependencies and base tools
# Stage 2: Deps - R packages (cached layer)
# Stage 3: Runtime - Final application build

# NOTE: Any ARG used in a FROM instruction must be declared before the first FROM.
# Declare DEPS_IMAGE here so it can be substituted in the runtime stage's FROM.
# Default "deps" allows local multi-stage fallback when no external deps image is supplied.
ARG DEPS_IMAGE=deps
ARG BASE_IMAGE=rocker/r-base:4.4.2

# ============================================================================
# Stage 1: Base - System dependencies
# ============================================================================

# Build args used by workflows:
# - BASE_IMAGE: When the workflow finds a monthly base image, it builds the deps
#   stage with `--build-arg BASE_IMAGE=ghcr.io/<owner>/<repo>/base-deps:<TIME_BUCKET>`.
#   Otherwise this defaults to `rocker/r-base:4.4.2`.
# - SKIP_BASE_DEPS: Set to `true` by the workflow when using a prebuilt base image, so
#   the heavy apt-get install below is skipped. When building from scratch locally or
#   when no base image exists, leave as `false` to install system dependencies.
# - Cache hint: The workflow also provides `--cache-from <base-image>` when available
#   to accelerate this stage if it does run.

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG SKIP_BASE_DEPS=false

# Install system dependencies (skipped if using pre-built base image)
# To use a pre-built base: --build-arg BASE_IMAGE=ghcr.io/.../base-deps:tag --build-arg SKIP_BASE_DEPS=true
RUN if [ "$SKIP_BASE_DEPS" = "false" ]; then \
    apt-get update && apt-get install -y \
        build-essential \
        libcurl4-openssl-dev \
        libssl-dev \
        uuid-dev \
        libxml2-dev \
        libgpgme11-dev \
        squashfs-tools \
        libseccomp-dev \
        r-cran-devtools \
        libsqlite3-dev \
        libgit2-dev \
        pkg-config \
        git-all \
        wget \
        libbz2-dev \
        zlib1g-dev \
        python3-dev \
        libffi-dev \
        libfontconfig1-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libfreetype6-dev \
        libpng-dev \
        libtiff5-dev \
        libjpeg-dev \
        libmbedtls-dev \
        cargo \
        libmagick++-dev \
        libudunits2-dev \
        libgsl-dev \
        libtbb-dev \
        cmake \
        libcairo2-dev \
        libgpg-error-dev \
        libgmp-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libnode-dev \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Skipping base dependency installation (using pre-built base image)"; \
    fi

# ============================================================================
# Stage 2: Deps - Install R dependencies
# ============================================================================

# Build args and control flow:
# - This stage is built when producing the dependency image (`--target deps`).
# - During runtime builds, if the workflow passes an external deps image via
#   `--build-arg DEPS_IMAGE=ghcr.io/<owner>/<repo>/deps:<HASH>-<TIME_BUCKET>` and
#   targets the runtime stage, Docker will not build this `deps` stage at all.
# - Fallback: If no DEPS_IMAGE is supplied, DEPS_IMAGE defaults to `deps`, so
#   `FROM deps` in the runtime stage will resolve to this local stage; Docker will
#   build Base -> Deps first, then proceed to Runtime.
# - GH_PAT: Optional token for R installs. If provided by callers, it is honored here.
FROM base AS deps

ARG DEBIAN_FRONTEND=noninteractive
ARG GH_PAT='NOT_SET'

# Install R dependencies
RUN apt-get update && apt-get install -y r-base r-base-dev && \
    if [ "${GH_PAT}" != 'NOT_SET' ]; then \
        echo 'Setting GH_PAT'; \
        export GITHUB_PAT="${GH_PAT}"; \
    fi && \
    Rscript -e "install.packages(c('remotes', 'devtools', 'BiocManager', 'pryr', 'rmdformats', 'knitr', 'logger', 'Matrix', 'kernlab', 'tidyverse', 'leidenbase', 'igraph', 'FNN', 'plyr'), lib='/usr/local/lib/R/site-library', dependencies=TRUE, ask = FALSE)" && \
    echo "local({options(repos = BiocManager::repositories())})" >> ~/.Rprofile && \
    Rscript -e "BiocManager::install(c('DESeq2', 'HDF5Array', 'DelayedArray', 'ComplexHeatmap'), ask = FALSE, update = TRUE)" && \
    Rscript -e "install.packages(c('SeuratObject', 'pscl', 'methods', 'parallel', 'mgcv', 'stats', 'testthat', 'Seurat', 'future', 'future.apply', 'progressr', 'rmarkdown', 'umap', 'ggrepel', 'pheatmap', 'ggplot2', 'circlize', 'viridisLite'), lib='/usr/local/lib/R/site-library', dependencies=TRUE, ask = FALSE)" && \
    rm -rf /var/lib/apt/lists/* /tmp/downloaded_packages/ /tmp/*.rds

# ============================================================================
# Stage 3: Runtime - Build and install PreGraphModeling package
# ============================================================================
# Build args and control flow:
# - DEPS_IMAGE (declared before any FROM):
#   * Workflow runtime build passes `--build-arg DEPS_IMAGE=<deps image tag>` and
#     `--target runtime`. Docker resolves `FROM ${DEPS_IMAGE}` and skips building
#     Base/Deps stages entirely.
#   * Local fallback: If DEPS_IMAGE is unset, it defaults to `deps` and the runtime
#     stage inherits from the locally built `deps` stage (Base -> Deps will be built
#     as needed).
# - Intent: Keep all heavy dependency installation in the deps image; the runtime
#   stage should only add application code and minimal work.
# - Testing note: In workflows where the runtime image is skipped, tests run against
#   the deps image directly; this Dockerfile remains compatible with that flow.

FROM ${DEPS_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG GH_PAT='NOT_SET'

# Copy application code
ADD . /PreGraphModeling

# Set working directory so tests running with '.' can locate DESCRIPTION
WORKDIR /PreGraphModeling

# Build and install PreGraphModeling
# upgrade = 'never' is specified because the dependencies install/upgrading is already handled in the deps stage
RUN cd /PreGraphModeling && \
    R CMD build . && \
    R CMD INSTALL --build *.tar.gz && \
    rm -Rf /tmp/downloaded_packages/ /tmp/*.rds

CMD ["R"]
