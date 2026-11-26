# Every target delegates to scripts/build.sh, which is the single source of
# truth for what each one does — and which works on a machine with no make.
#
#   make            list the targets
#   make test       run the whole suite
#   make e2e        run the end-to-end suite in containers (needs only Docker)
#   make e2e-local  the same tests, using the toolchain on this machine
#   make bin        release binary into ./bin
#   make bin-arm64  cross-compile for aarch64 (uConsole) into ./bin; also
#                   bin-/deb-/rpm- for armhf, riscv64 and static (musl x86-64)
#   make bin-freebsd / pkg-freebsd  FreeBSD binary / `pkg add` package, built
#                   in a container from Linux (needs only Docker)
#   make install    install under PREFIX (default /usr/local)
#
# Variables are passed through: make install PREFIX=$HOME/.local
#                               make run ARGS="--refresh 500 --tree"

BUILD := ./scripts/build.sh

# Arguments for `make run`.
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help build release bin run test test-unit e2e e2e-local check fmt fmt-check \
        lint ci deb rpm dist install uninstall snapshot clean version \
        bin-arm64 bin-armhf bin-riscv64 bin-static deb-arm64 deb-armhf \
        deb-riscv64 deb-static rpm-arm64 rpm-armhf rpm-riscv64 rpm-static \
        bin-freebsd pkg-freebsd

help:
	@$(BUILD) help

build:
	@$(BUILD) build

release:
	@$(BUILD) release

bin:
	@$(BUILD) bin

run:
	@$(BUILD) run $(ARGS)

test:
	@$(BUILD) test

test-unit:
	@$(BUILD) test-unit

e2e:
	@$(BUILD) e2e

e2e-local:
	@$(BUILD) e2e-local

check:
	@$(BUILD) check

fmt:
	@$(BUILD) fmt

fmt-check:
	@$(BUILD) fmt-check

lint:
	@$(BUILD) lint

ci:
	@$(BUILD) ci

deb:
	@$(BUILD) deb

rpm:
	@$(BUILD) rpm

bin-arm64:
	@$(BUILD) bin-arm64

bin-armhf:
	@$(BUILD) bin-armhf

bin-riscv64:
	@$(BUILD) bin-riscv64

bin-static:
	@$(BUILD) bin-static

bin-freebsd:
	@$(BUILD) bin-freebsd

pkg-freebsd:
	@$(BUILD) pkg-freebsd

deb-arm64:
	@$(BUILD) deb-arm64

deb-armhf:
	@$(BUILD) deb-armhf

deb-riscv64:
	@$(BUILD) deb-riscv64

deb-static:
	@$(BUILD) deb-static

rpm-arm64:
	@$(BUILD) rpm-arm64

rpm-armhf:
	@$(BUILD) rpm-armhf

rpm-riscv64:
	@$(BUILD) rpm-riscv64

rpm-static:
	@$(BUILD) rpm-static

dist:
	@$(BUILD) dist

install:
	@$(BUILD) install

uninstall:
	@$(BUILD) uninstall

snapshot:
	@$(BUILD) snapshot

clean:
	@$(BUILD) clean

version:
	@$(BUILD) version
