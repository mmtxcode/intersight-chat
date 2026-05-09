SHELL := /bin/bash
COMPOSE ?= docker compose
MODEL ?= llama3.1

.PHONY: up up-gpu down restart logs ps build pull-model shell-app shell-ollama clean

## Bring the stack up (CPU). Works on any Docker host. First run also builds the app image.
up:
	$(COMPOSE) up -d --build

## Bring the stack up with NVIDIA GPU passthrough for Ollama.
## Requires the NVIDIA Container Toolkit on the host.
up-gpu:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

## Run only the app container; reuse an Ollama instance already on the host.
## Use this when port 11434 is taken because Ollama is running natively.
up-host-ollama:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.host-ollama.yml up -d --build app

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
