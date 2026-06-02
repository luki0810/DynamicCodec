#!/usr/bin/env bash
# Bring the dyc_luki container into a state where main.py / train.py just work.
#
# Two image paths are supported:
#
#   1. Tencent internal mirror (default; fast):
#        mirrors.tencent.com/dct/facodec_lukilu:v1.2
#      Pulled if not present locally. Requires Tencent network access.
#
#   2. Public-friendly local build (fallback / external collaborators):
#        dynamiccodec:local
#      Built from .docker/Dockerfile, based on pytorch/pytorch:2.6.0-cuda12.4
#      from Docker Hub.
#
# The script tries the Tencent image first. If it cannot be pulled (no network
# access to the mirror), it falls back to building the local image. You can
# also force one path explicitly:
#
#   bash scripts/setup_container.sh                  # auto: try Tencent, fallback to public
#   bash scripts/setup_container.sh --public         # force the public local build
#   bash scripts/setup_container.sh --image my:tag   # use whatever image you've prepared
#   bash scripts/setup_container.sh --rm             # tear down container first
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
#        - default                 → try Tencent image; if unreachable, build local
#   4. Reuse-or-create the dyc_luki container with the project + /sec-cfs-nj mounted.
#   5. pip-install the project's pip dependencies (no-op if image already has them).
#   6. Apply argbind / audiotools source patches (idempotent).
#   7. Run smoke check.
#   8. Reclaim ownership of files written into runs/ by past root-owned processes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TENCENT_IMAGE="mirrors.tencent.com/dct/facodec_lukilu:v1.2"
LOCAL_IMAGE="dynamiccodec:local"
NAME="dyc_luki"

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
    # Auto path: prefer Tencent, fall back to local public build.
    IMAGE="$TENCENT_IMAGE"
    if have_image_locally "$IMAGE"; then
        echo "  using local copy of $IMAGE"
    else
        echo "  $IMAGE not local; probing Tencent mirror..."
        if try_pull "$IMAGE"; then
            echo "  pulled from Tencent mirror"
        else
            echo "  Tencent mirror unreachable, falling back to public build"
            IMAGE="$LOCAL_IMAGE"
            if ! have_image_locally "$IMAGE"; then
                build_local_image
            fi
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
    # Mount /sec-cfs-nj only if it exists on the host (external machines won't have it).
    cfs_mount=()
    if [ -d /sec-cfs-nj ]; then
        cfs_mount=(-v /sec-cfs-nj:/sec-cfs-nj)
    else
        echo "  note: /sec-cfs-nj not present on host; skipping that mount"
    fi
    $DK run -d \
        -v "$REPO_ROOT":/app \
        "${cfs_mount[@]}" \
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
