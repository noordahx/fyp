.PHONY: clean zip help

help:
	@echo " make clean		- Remove __pycache__ and other temp files"
	@echo " make zip		- Create a zip file"
	@echo " make help 		- Show this help message"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Clean up complete"

zip: clean
	@echo "Creating zip file of the project for GPU server deployment"
	zip -r fyp_project.zip \
		configs/ \
		scripts/ \
		src/ \
		README.md \
		requirements.txt \
		Makefile \
		run_complete_evaluation.py \
		test_evaluation_pipeline.py \
		COMPLETE_EVALUATION_GUIDE.md \
		CLAUDE.md \
		-x "*.git*" \
		-x "*__pycache__*" \
		-x "*.ipynb_checkpoints*" \
		-x "*.DS_Store" \
		-x "data/*" \
		-x "results/*" \
		-x "colab_results/*" \
		-x "ignore/*" \
		-x "*.pt" \
		-x "*.pth" \
		-x "*.log"
	@echo "Zip file created: fyp_project.zip"
	@echo ""
	@echo "📦 Ready for GPU server deployment!"
	@echo "📋 Deployment instructions:"
	@echo "  1. Transfer fyp_project.zip to your GPU server"
	@echo "  2. Extract: unzip fyp_project.zip"
	@echo "  3. Install dependencies: pip install -r requirements.txt"
	@echo "  4. Run pipeline test: python test_evaluation_pipeline.py"
	@echo "  5. Start evaluation: python run_complete_evaluation.py --config configs/evaluation_config.yaml"