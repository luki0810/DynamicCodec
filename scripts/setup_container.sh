#!/usr/bin/env bash
# Bring the dyc_dev container into a state where main.py / train.py just work.
#
# Two image paths are supported:
#
#   1. Private/internal pre-built image (optional, fast):
#        Configured via the DYC_INTERNAL_IMAGE env var, e.g.
#          export DYC_INTERNAL_IMAGE=registry.example.com/team/dyc:v1
#        Pulled if not present locally. Requires network access to that registry.
#
#   2. Public local build (default / external collaborators):
#        dynamiccodec:local
#      Built from .docker/Dockerfile, based on pytorch/pytorch:2.6.0-cuda12.4
#      from Docker Hub.
#
# If DYC_INTERNAL_IMAGE is set, the script tries it first and falls back to the
# public build when the pull fails. If unset, it goes straight to the public
# build. You can also force the path explicitly:
#
#   bash scripts/setup_container.sh                  # auto (env-driven)
#   bash scripts/setup_container.sh --public         # force the public local build
#   bash scripts/setup_container.sh --image my:tag   # use whatever image you've prepared
#   bash scripts/setup_container.sh --rm             # tear down container first
#
# Extra mounts:
#   DYC_DATA_MOUNT=/host/path             → mount as /host/path in the container
#   DYC_DATA_MOUNT=/host/path:/in/ctr     → explicit src:dst form
#
# Idempotent — re-running re-uses an existing healthy container, skips already-
# installed pip packages, and the patch scripts no-op when already applied.
#
# What the script does, in order:
#   1. Confirm we have docker access (via group membership or passwordless sudo).
#   2. (one-time) Add the current user to the `docker` group.
#   3. Decide which image to use:
#        - explicit --image <tag>  → use it as-is
#        - explicit --public       → build .docker/Dockerfile → dynamiccodec:local
#        - default                 → DYC_INTERNAL_IMAGE if set, else build local
#   4. Reuse-or-create the dyc_dev container (with optional DYC_DATA_MOUNT).
#   5. pip-install the project's pip dependencies (no-op if image already has them).
#   6. Apply argbind / audiotools source patches (idempotent).
#   7. Run smoke check.
#   8. Reclaim ownership of files written into runs/ by past root-owned processes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTERNAL_IMAGE="${DYC_INTERNAL_IMAGE:-}"
LOCAL_IMAGE="dynamiccodec:local"
NAME="dyc_dev"

# ---- arg parsing ----
REMOVE_FIRST=false
FORCE_PUBLIC=false
EXPLICIT_IMAGE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --rm) REMOVE_FIRST=true ;;
        --public) FORCE_PUBLIC=true ;;
        --image)
            shift
            [ $# -gt 0 ] || { echo "--image needs a tag" >&2; exit 2; }
            EXPLICIT_IMAGE="$1"
            ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

# ---- pick docker invocation ----
# Prefer unprivileged `docker` if the current shell is in the docker group
# AND can talk to the daemon; otherwise fall back to sudo.
if id -nG "$USER" | tr ' ' '\n' | grep -qx docker && docker info >/dev/null 2>&1; then
    DK="docker"
else
    DK="sudo docker"
fi

step() { printf '\n[%s] %s\n' "$1" "$2"; }

step 1/8 "verify docker access"
sudo -n true 2>/dev/null || { echo "  no passwordless sudo"; exit 1; }
$DK info >/dev/null 2>&1 || { echo "  cannot talk to docker daemon"; exit 1; }
echo "  using: $DK"

step 2/8 "ensure $USER is in docker group (one-time, takes effect next login)"
if getent group docker | awk -F: '{print $4}' | tr ',' '\n' | grep -qx "$USER"; then
    echo "  already in docker group"
else
    sudo usermod -aG docker "$USER"
    echo "  added — new shells will pick it up automatically"
fi

# ---- decide on image ----
have_image_locally() {
    $DK image inspect "$1" >/dev/null 2>&1
}

try_pull() {
    # Only used as a probe when the image isn't already local.
    $DK pull "$1" >/dev/null 2>&1
}

build_local_image() {
    echo "  building $LOCAL_IMAGE from .docker/Dockerfile (this can take ~10 min the first time)..."
    $DK build -t "$LOCAL_IMAGE" -f "$REPO_ROOT/.docker/Dockerfile" "$REPO_ROOT/.docker/" >/dev/null
}

step 3/8 "pick image"
if [ -n "$EXPLICIT_IMAGE" ]; then
    IMAGE="$EXPLICIT_IMAGE"
    echo "  using explicit --image: $IMAGE"
    if ! have_image_locally "$IMAGE"; then
        echo "  not found locally; attempting pull..."
        try_pull "$IMAGE" || { echo "  pull failed for $IMAGE"; exit 1; }
    fi
elif $FORCE_PUBLIC; then
    IMAGE="$LOCAL_IMAGE"
    echo "  --public: using $IMAGE"
    if ! have_image_locally "$IMAGE"; then
        build_local_image
    else
        echo "  reusing existing local image"
    fi
else
    # Auto path: prefer the configured internal image (if any), fall back to public build.
    if [ -n "$INTERNAL_IMAGE" ]; then
        IMAGE="$INTERNAL_IMAGE"
        if have_image_locally "$IMAGE"; then
            echo "  using local copy of $IMAGE"
        else
            echo "  $IMAGE not local; probing internal registry..."
            if try_pull "$IMAGE"; then
                echo "  pulled from internal registry"
            else
                echo "  internal registry unreachable, falling back to public build"
                IMAGE="$LOCAL_IMAGE"
                if ! have_image_locally "$IMAGE"; then
                    build_local_image
                fi
            fi
        fi
    else
        IMAGE="$LOCAL_IMAGE"
        echo "  DYC_INTERNAL_IMAGE not set; using public build $IMAGE"
        if ! have_image_locally "$IMAGE"; then
            build_local_image
        else
            echo "  reusing existing local image"
        fi
    fi
fi
echo "  → image=$IMAGE"

step 4/8 "container '$NAME'"
if $REMOVE_FIRST; then
    if $DK ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
        echo "  --rm: removing existing container"
        $DK rm -f "$NAME" >/dev/null
    fi
fi

if $DK ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    if $DK ps --format '{{.Names}}' | grep -qx "$NAME"; then
        echo "  already running"
    else
        $DK start "$NAME" >/dev/null
        echo "  started existing container"
    fi
else
    # Optional extra mount via DYC_DATA_MOUNT env var.
    # Accepts either "/host/path" (mounted at the same path inside the container)
    # or "/host/path:/in/container" for an explicit src:dst pair.
    extra_mount=()
    if [ -n "${DYC_DATA_MOUNT:-}" ]; then
        case "$DYC_DATA_MOUNT" in
            *:*) src="${DYC_DATA_MOUNT%%:*}" ;;
              *) src="$DYC_DATA_MOUNT" ;;
        esac
        if [ -d "$src" ]; then
            extra_mount=(-v "$DYC_DATA_MOUNT")
            echo "  mounting DYC_DATA_MOUNT=$DYC_DATA_MOUNT"
        else
            echo "  note: DYC_DATA_MOUNT=$DYC_DATA_MOUNT does not exist on host; skipping"
        fi
    fi
    $DK run -d \
        -v "$REPO_ROOT":/app \
        "${extra_mount[@]}" \
        -w /app \
        --gpus all \
        --network host \
        --name "$NAME" \
        "$IMAGE" \
        tail -f /dev/null >/dev/null
    echo "  created"
fi

step 5/8 "install / verify pip packages"
# Safe to re-run on the public image too — pip skips already-installed packages.
$DK exec "$NAME" pip install --no-cache-dir --quiet \
    descript-audiotools==0.7.2 \
    argbind==0.3.9 \
    vocos==0.1.0 \
    typeguard \
    humanfriendly

# SSL feature path (data2vec / hubert / whisper) needs fairseq + openai-whisper.
# fairseq 0.12.2 pins omegaconf<2.1, whose metadata fails newer pip's PEP 440
# validation. We pin pip to 24.0 just for these installs, then restore.
echo "  installing SSL deps (fairseq / whisper / omegaconf 2.0.x)"
ORIG_PIP=$($DK exec "$NAME" pip --version | awk '{print $2}')
$DK exec "$NAME" pip install --no-cache-dir --quiet 'pip==24.0'
$DK exec "$NAME" pip install --no-cache-dir --quiet --no-deps \
    fairseq==0.12.2 \
    omegaconf==2.0.6 \
    hydra-core==1.0.7
$DK exec "$NAME" pip install --no-cache-dir --quiet \
    antlr4-python3-runtime==4.8 \
    bitarray \
    sacrebleu
# tiktoken needs prebuilt wheel (no Rust in image)
$DK exec "$NAME" pip install --no-cache-dir --quiet --only-binary :all: tiktoken
$DK exec "$NAME" pip install --no-cache-dir --quiet openai-whisper
# restore pip
$DK exec "$NAME" pip install --no-cache-dir --quiet "pip==$ORIG_PIP" || true
echo "  done"

step 6/8 "apply patches"
$DK exec "$NAME" python /app/scripts/patch_argbind.py
$DK exec "$NAME" python /app/scripts/patch_audiotools_torchload.py

step 7/8 "smoke check"
$DK exec "$NAME" python /app/scripts/smoke_check_container.py

step 8/8 "reclaim ownership of runs/ from past root-owned container writes"
if [ -d "$REPO_ROOT/runs" ]; then
    if find "$REPO_ROOT/runs" -not -user "$USER" -print -quit | grep -q .; then
        echo "  found root-owned files; chowning to $USER"
        sudo chown -R "$USER":"$USER" "$REPO_ROOT/runs"
    else
        echo "  nothing to do"
    fi
fi

echo
echo "✓ container ready (image: $IMAGE)"
echo "  enter:  $DK exec -it $NAME bash"
echo "  infer:  $DK exec $NAME bash -c 'cd /app && python main.py --conf_path conf/base.yaml --save_path runs/inference_dac --args.debug 1'"
