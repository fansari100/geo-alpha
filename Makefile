PYTHON ?= python3
COMPOSE ?= docker compose

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

.PHONY: install test smoke
install:  ## install python deps for quant + api
	$(PYTHON) -m pip install -e quant[dev] -e api[dev]

test:  ## run python tests
	cd quant && pytest -v
	cd api   && pytest -v

smoke: ## run the end-to-end smoke
	$(PYTHON) scripts/smoke.py

.PHONY: cpp-build cpp-test cpp-bench
cpp-build:  ## build C++ engine
	cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release
	cmake --build engine/build --parallel

cpp-test: cpp-build
	ctest --test-dir engine/build --output-on-failure

cpp-bench: cpp-build
	./engine/build/geoalpha_bench --steps 100000

.PHONY: web-install web-dev web-build
web-install:
	cd web && npm install --no-audit --no-fund

web-dev: web-install
	cd web && npm run dev

web-build: web-install
	cd web && npm run build

.PHONY: java-build
java-build:
	cd service && mvn -B -ntp verify

.PHONY: up down logs
up:    ## bring up the whole stack
	$(COMPOSE) up --build -d
down:
	$(COMPOSE) down -v
logs:
	$(COMPOSE) logs -f
