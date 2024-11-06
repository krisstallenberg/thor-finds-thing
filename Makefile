ENV_DIR := ./myenv
REQUIREMENTS := requirements.txt

# Set up the Conda environment and install dependencies
.PHONY: setup
setup:
	conda create -y -p $(ENV_DIR) python=3.11
	conda run -p $(ENV_DIR) pip3 install -r $(REQUIREMENTS)

# Run the main.py script in the Conda environment
.PHONY: run
run:
	conda run -p $(ENV_DIR) python3 main.py

# Run jupyter lab in the Conda environment
.PHONY: jupyter
jupyter:
	conda run -p $(ENV_DIR) jupyter lab

# Clean up the Conda environment
.PHONY: clean
clean:
	conda remove -y -p $(ENV_DIR) --all

# Instructions for activating the environment
.PHONY: activate
activate:
	@echo "Run 'conda activate $(ENV_DIR)' to activate the environment."