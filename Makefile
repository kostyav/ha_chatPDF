COMPOSE  = docker compose -f src/part2/docker-compose.yml
IMG_DIR  = .docker_images
IMAGES   = rag-part2-parser rag-part2-text_indexer rag-part2-visual_indexer rag-part2-orchestrator

# Build all images and save them to the persistent workspace
build:
	$(COMPOSE) build
	mkdir -p $(IMG_DIR)
	@for img in $(IMAGES); do \
		echo "Saving $$img …"; \
		docker save $$img -o $(IMG_DIR)/$$img.tar; \
	done
	@echo "Images saved to $(IMG_DIR)/"

# Load pre-built images (fast — no rebuild) then start
up:
	@if ls $(IMG_DIR)/*.tar 1>/dev/null 2>&1; then \
		echo "Loading cached images …"; \
		for f in $(IMG_DIR)/*.tar; do docker load -i $$f; done; \
		$(COMPOSE) up; \
	else \
		echo "No cached images found — running full build …"; \
		$(MAKE) build; \
		$(COMPOSE) up; \
	fi

# Pull infrastructure images (redis, qdrant, ollama) into the cache too
pull:
	docker pull redis:7-alpine
	docker pull qdrant/qdrant:latest
	docker pull ollama/ollama:latest
	mkdir -p $(IMG_DIR)
	docker save redis:7-alpine      -o $(IMG_DIR)/redis.tar
	docker save qdrant/qdrant:latest -o $(IMG_DIR)/qdrant.tar
	docker save ollama/ollama:latest -o $(IMG_DIR)/ollama.tar

# Load ALL saved images (infra + services)
load:
	@for f in $(IMG_DIR)/*.tar; do \
		echo "Loading $$f …"; \
		docker load -i $$f; \
	done

down:
	$(COMPOSE) down

.PHONY: build up pull load down
