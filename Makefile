SHELL := /bin/bash
COMPOSE ?= docker compose
MODEL ?= qwen2.5:14b

.PHONY: up up-gpu down restart logs ps build pull-model shell-app shell-ollama clean

## Bring the stack up (CPU only). Works on any Docker host. For GPU servers,
## use `make up-gpu` instead.
up:
	$(COMPOSE) up -d --build

## Bring the stack up with NVIDIA GPU passthrough for Ollama.
## Requires the NVIDIA Container Toolkit on the host. This is the recommended
## target for production / lab GPU servers.
up-gpu:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

## Stop and remove containers (keeps the model volume).
down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart

logs:
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

build:
	$(COMPOSE) build

## Pull a model into the running ollama container. Override with MODEL=qwen2.5 etc.
pull-model:
	$(COMPOSE) exec ollama ollama pull $(MODEL)

shell-app:
	$(COMPOSE) exec app bash

shell-ollama:
	$(COMPOSE) exec ollama bash

## Full teardown including downloaded models. Use with care.
clean:
	$(COMPOSE) down -v
