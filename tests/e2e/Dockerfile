# The image `tests/e2e_docker.rs` drives.
#
# What "hermetic" means here, exactly:
#
#   * The test run reaches nothing. Every container gets an empty network
#     namespace, a read-only image and a tmpfs for scratch, so a test cannot
#     depend on the network, on the machine running the suite, or on anything
#     a previous test left behind. `tests/e2e_docker.rs` asserts all of that
#     rather than assuming it.
#   * The image is reproducible. Both bases are pinned by digest and the
#     dependency set is pinned by `Cargo.lock` with `--locked`, so the same
#     checkout builds the same image.
#   * The image *build* still fetches those crates from crates.io, which is
#     the one input that needs the network. `cargo vendor` would close it, at
#     the cost of 446 MB in the repository — 44 MB if the Windows crates this
#     image never compiles are dropped — which is not a trade worth making for
#     a test suite. Pinning fixes what is fetched; it does not remove the
#     fetch.
#
# Two stages so the thing under test is built from this checkout rather than
# copied in from the host: a binary built against the host's glibc and run on
# some other distribution's is not the binary anyone ships. Both bases are
# pinned by digest, so the container a failure reproduces in is byte for byte
# the one it was found in — a tag like `bookworm-slim` is republished, and a
# test that silently changes its own operating system between runs is not a
# test of anything.
#
# The runtime stage is deliberately bare. crabmon reads /proc and nothing else,
# and a container with almost no processes in it is exactly the controlled
# machine an end-to-end assertion wants: `pid:1` means one process, and a
# workload started by the test is the only thing moving.
# rust:1.90-slim-bookworm
FROM rust@sha256:64232e656c058f4468e8d024e990acff04f0fd5a5c0a88a574dc37773d7325c9 AS build
WORKDIR /src
# Only what the build reads. The test suite is not copied: these tests run the
# binary from outside, and `cargo test` inside the image would be the unit
# suite over again on a machine that is harder to debug.
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked --bin crabmon

# debian:bookworm-slim
FROM debian@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251
# Nothing is installed. The exporter is scraped from inside the container,
# which used to mean `apt-get install curl` — a package fetched from whatever
# the mirrors held that day, which is a large uncontrolled input to a test
# whose whole claim is that it is hermetic. `httpc` does the one request that
# is needed with bash's `/dev/tcp`, which the base image already has.
COPY tests/e2e/httpc /usr/local/bin/httpc
COPY --from=build /src/target/release/crabmon /usr/local/bin/crabmon
# A writable place for recordings and exports that is not the image.
WORKDIR /work
ENTRYPOINT ["crabmon"]
