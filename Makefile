# Every target delegates to scripts/build.sh, which is the single source of
# truth for what each one does — and which works on a machine with no make.
#
#   make            list the targets
#   make test       run the whole suite
#   make e2e        run the end-to-end suite in containers (needs only Docker)
#   make e2e-local  the same tests, using the toolchain on this machine
#   make bin        release binary into ./bin
#   make bin-arm64  cross-compile for aarch64 (uConsole) into ./bin
#   make install    install under PREFIX (default /usr/local)
#
# Variables are passed through: make install PREFIX=$HOME/.local
#                               make run ARGS="--refresh 500 --tree"

BUILD := ./scripts/build.sh

# Arguments for `make run`.
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help build release bin bin-arm64 bin-armhf deb-armhf run test test-unit e2e e2e-local check fmt fmt-check \
        lint ci deb deb-arm64 dist install uninstall snapshot clean version

help:
	@$(BUILD) help

build:
	@$(BUILD) build

release:
	@$(BUILD) release

bin:
	@$(BUILD) bin

bin-arm64:
	@$(BUILD) bin-arm64

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

deb-armhf:
	@$(BUILD) deb-armhf

bin-armhf:
	@$(BUILD) bin-armhf

deb-arm64:
	@$(BUILD) deb-arm64

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
