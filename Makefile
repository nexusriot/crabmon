# Every target delegates to scripts/build.sh, which is the single source of
# truth for what each one does — and which works on a machine with no make.
#
#   make            list the targets
#   make test       run the whole suite
#   make bin        release binary into ./bin
#   make install    install under PREFIX (default /usr/local)
#
# Variables are passed through: make install PREFIX=$HOME/.local
#                               make run ARGS="--refresh 500 --tree"

BUILD := ./scripts/build.sh

# Arguments for `make run`.
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help build release bin run test test-unit check fmt fmt-check lint ci \
        deb dist install uninstall clean version

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

dist:
	@$(BUILD) dist

install:
	@$(BUILD) install

uninstall:
	@$(BUILD) uninstall

clean:
	@$(BUILD) clean

version:
	@$(BUILD) version
