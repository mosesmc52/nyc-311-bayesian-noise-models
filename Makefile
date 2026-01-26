# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PYTHON ?= python3

DATA_DIR := data
LOOKUP_CSV := $(DATA_DIR)/nyc_county_fips.csv

COUNTY_SCRIPT := scripts/geographies/download_nyc_county_shapes.py
NOISE_SCRIPT  := scripts/ingest/download_nyc_311_noise.py

# -------------------------------------------------------------------
# Phony targets
# -------------------------------------------------------------------

.PHONY: help all geos noise clean

# -------------------------------------------------------------------
# Help
# -------------------------------------------------------------------

help:
	@echo "Available targets:"
	@echo "  make geos    - Download NYC county shapefiles and write GeoJSON"
	@echo "  make noise   - Download NYC 311 noise data"
	@echo "  make all     - Run geos then noise"
	@echo "  make clean   - Remove derived data outputs"

# -------------------------------------------------------------------
# Main targets
# -------------------------------------------------------------------

all: geos noise

geos:
	@echo "==> Downloading NYC county shapefiles"
	$(PYTHON) $(COUNTY_SCRIPT) \
		--csv $(LOOKUP_CSV) \
		--out-dir $(DATA_DIR) \
		--per-borough

noise:
	@echo "==> Downloading NYC 311 noise data"
	$(PYTHON) $(NOISE_SCRIPT)

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------

clean:
	@echo "==> Cleaning data directory (excluding data/lookups)"
	@find data -mindepth 1 \
		-not -path "data/lookups*" \
		-not -path "data/lookups/*" \
		-exec rm -rf {} +
