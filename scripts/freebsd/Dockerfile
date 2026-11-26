# A hermetic FreeBSD cross toolchain, so `make bin-freebsd` needs Docker and
# nothing else — no FreeBSD host, no VM, no rustup target on the machine.
#
# Rust has a FreeBSD target, but linking for it needs FreeBSD's libc and
# crt files. Those come from the release's own `base.txz`, pinned by SHA-256
# below, and only the three directories a link needs are kept. The linker is
# the distribution's clang with lld; nothing is compiled by a FreeBSD tool.
#
# The binary is built against FreeBSD 15.0 and runs on later releases: the
# kernel and libc keep old ABIs. It is the *older* side that does not work,
# so a 14.x host needs `FREEBSD_RELEASE` lowered and the image rebuilt.

# rust:1.95-slim-bookworm — the same pin as tests/e2e/Dockerfile.runner.
FROM rust@sha256:d7482085ff5b415f84dba5647ae71606650bdef00db7aeb69f4b3d170c3e4082

ARG FREEBSD_RELEASE=15.0
ARG BASE_SHA256=ac0c933cc02ee8af4da793f551e4a9a15cdcf0e67851290b1e8c19dd6d30bba8

RUN apt-get update \
 && apt-get install -y --no-install-recommends clang lld curl ca-certificates xz-utils \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/freebsd-sysroot \
 && curl -fsSL -o /tmp/base.txz \
      "https://download.freebsd.org/releases/amd64/amd64/${FREEBSD_RELEASE}-RELEASE/base.txz" \
 && echo "${BASE_SHA256}  /tmp/base.txz" | sha256sum -c - \
 && tar -xJf /tmp/base.txz -C /opt/freebsd-sysroot ./lib ./usr/lib ./usr/include \
 && rm /tmp/base.txz

RUN rustup target add x86_64-unknown-freebsd

# `-lgcc_s` and friends live in the sysroot; the wrapper only has to say where.
RUN printf '#!/bin/sh\nexec clang --target=x86_64-unknown-freebsd%s --sysroot=/opt/freebsd-sysroot -fuse-ld=lld "$@"\n' \
      "${FREEBSD_RELEASE%%.*}" > /usr/local/bin/freebsd-cc \
 && chmod +x /usr/local/bin/freebsd-cc

ENV CARGO_TARGET_X86_64_UNKNOWN_FREEBSD_LINKER=/usr/local/bin/freebsd-cc \
    CARGO_HOME=/cargo \
    CARGO_TARGET_DIR=/target
WORKDIR /src
